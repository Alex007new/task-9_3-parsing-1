import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response
from requests.exceptions import RequestException
from sqlalchemy import create_engine
# 
# 1) Конфигурация парсера
# 
@dataclass(frozen=True)
class ParserConfig:
    base_url: str
    headers: Dict[str, str]
    delay_seconds: Tuple[float, float]  # (min_delay, max_delay)
    max_retries: int
    timeout_seconds: int
    retry_statuses: Tuple[int, ...] = (403, 408, 429, 500, 502, 503, 504)


DEFAULT_CONFIG = ParserConfig(
    base_url="https://books.toscrape.com/",
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    },
    delay_seconds=(1.5, 3.5),
    max_retries=5,
    timeout_seconds=20,
)
# 
# 2) HTTP-клиент с ретраями
# 
def sleep_human(min_max: Tuple[float, float]) -> None:
    """Небольшая случайная задержка, имитирующая человека."""
    time.sleep(random.uniform(min_max[0], min_max[1]))


def fetch_with_retries(
    url: str,
    config: ParserConfig,
    session: requests.Session,
) -> Optional[Response]:
    """
    Делает GET с повторами при сетевых ошибках и "плохих" статусах.
    Возвращает Response или None, если все попытки провалились.
    """
    for attempt in range(1, config.max_retries + 1):
        try:
            response = session.get(
                url,
                headers=config.headers,
                timeout=config.timeout_seconds,
            )

            if response.status_code == 200:
                return response

            # Ретраим только те статусы, которые обычно временные/антибот
            if response.status_code in config.retry_statuses:
                print(
                    f"[retry] {url} -> status={response.status_code}, "
                    f"attempt={attempt}/{config.max_retries}"
                )
                sleep_human(config.delay_seconds)
                continue

            # Для остальных статусов — считаем фатальным для этой страницы
            print(f"[fail] {url} -> status={response.status_code} (no retry)")
            return None

        except RequestException as exc:
            # Сетевые проблемы, DNS, timeout, connection reset, etc.
            print(
                f"[error] {url} -> {type(exc).__name__}: {exc}, "
                f"attempt={attempt}/{config.max_retries}"
            )
            sleep_human(config.delay_seconds)
            continue

    return None
# 
# 3) Парсинг одной страницы каталога
# 
def parse_books_from_catalog_page(html: str, page_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    book_cards = soup.select("article.product_pod")

    rows: List[Dict] = []
    for card in book_cards:
        title = card.h3.a.get("title", "").strip()

        # Относительная ссылка на страницу книги
        rel_link = card.h3.a.get("href", "").strip()
        product_url = urljoin(page_url, rel_link)

        # Цена
        price_text = card.select_one("p.price_color")
        price = price_text.get_text(strip=True) if price_text else ""

        # Наличие
        availability_tag = card.select_one("p.instock.availability")
        availability = availability_tag.get_text(" ", strip=True) if availability_tag else ""

        # Рейтинг: класс вида "star-rating Three"
        rating_tag = card.select_one("p.star-rating")
        rating = ""
        if rating_tag:
            classes = rating_tag.get("class", [])
            # первая часть = "star-rating", вторая = "One/Two/Three..."
            rating = next((c for c in classes if c != "star-rating"), "")

        rows.append(
            {
                "title": title,
                "price_raw": price,
                "availability": availability,
                "rating": rating,
                "product_url": product_url,
            }
        )

    # next page
    next_link_tag = soup.select_one("li.next > a")
    next_url = urljoin(page_url, next_link_tag["href"]) if next_link_tag else None

    return rows, next_url
# 
# 4) Пагинация: обходим весь каталог
# 
def scrape_all_books(config: ParserConfig) -> pd.DataFrame:
    session = requests.Session()

    current_url = urljoin(config.base_url, "catalogue/page-1.html")
    all_rows: List[Dict] = []
    page_num = 0

    while current_url:
        page_num += 1
        print(f"[page] {page_num}: {current_url}")

        response = fetch_with_retries(current_url, config, session)
        if response is None:
            print(f"[skip] failed to fetch page: {current_url}")
            break

        (rows, next_url) = parse_books_from_catalog_page(response.text, current_url)

        # добавим служебные поля
        for r in rows:
            r["page_num"] = page_num

        all_rows.extend(rows)

        # если на странице нет карточек — вероятно, бан/сломалась верстка
        if not rows:
            print("[stop] no books found on page -> stopping")
            break

        current_url = next_url
        sleep_human(config.delay_seconds)

    df = pd.DataFrame(all_rows)
    return df
# 
# 5) Очистка + проверки целостности + агрегации
# 
def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No data scraped: dataframe is empty")

    # Нормализуем цену: "£51.77" -> 51.77
    
    df["price_gbp"] = (
    df["price_raw"]
      .astype(str)
      .str.replace(r"[^\d.]", "", regex=True)  # удаляем всё, кроме цифр и точки
)

    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")

    # Простые проверки целостности
    checks = {
        "title_not_null": df["title"].notna().mean(),
        "product_url_not_null": df["product_url"].notna().mean(),
        "price_parsed_ratio": df["price_gbp"].notna().mean(),
    }

    print("[data_quality]")
    for k, v in checks.items():
        print(f"  {k}: {v:.3f}")

    # Дедупликация по ссылке
    df = df.drop_duplicates(subset=["product_url"]).reset_index(drop=True)

    # Агрегации
    agg_by_rating = (
        df.groupby("rating", dropna=False)
        .agg(
            books_count=("product_url", "count"),
            avg_price=("price_gbp", "mean"),
            min_price=("price_gbp", "min"),
            max_price=("price_gbp", "max"),
        )
        .sort_values("books_count", ascending=False)
    )

    print("\n[aggregation] by rating")
    print(agg_by_rating)

    return df
# 
# 6) Фильтрация + загрузка в Postgres
# 
def filter_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Пример фильтра: берем книги, которые в наличии и дешевле 30 GBP.
    """
    mask_in_stock = df["availability"].str.contains("In stock", case=False, na=False)
    mask_price = df["price_gbp"].notna() & (df["price_gbp"] < 30)
    filtered = df.loc[mask_in_stock & mask_price].copy()
    return filtered


def load_to_postgres(df: pd.DataFrame, pg_dsn: str, table_name: str = "books_filtered") -> None:
    """
    pg_dsn , типа postgresql+psycopg2://user:password@localhost:5432/mydb
    """
    if df.empty:
        print("[db] filtered dataframe is empty -> nothing to load")
        return

    engine = create_engine(pg_dsn)

    # заранее приводим колонки
    cols = [
        "title",
        "price_gbp",
        "availability",
        "rating",
        "product_url",
        "page_num",
    ]
    df_to_load = df[cols].copy()

    df_to_load.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"[db] loaded {len(df_to_load)} rows into {table_name}")
# 
# 7) Main
# 
def main() -> None:
    config = DEFAULT_CONFIG

    # 1) Парсим все страницы каталога
    df_raw = scrape_all_books(config)

    # 2) Сохраняем "сырые" данные
    df_raw.to_csv("books_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[csv] saved raw: {len(df_raw)} rows -> books_raw.csv")

    # 3) Чистим + проверки качества + агрегации
    df_clean = clean_and_validate(df_raw)
    df_clean.to_csv("books_clean.csv", index=False, encoding="utf-8-sig")
    print(f"[csv] saved clean: {len(df_clean)} rows -> books_clean.csv")

    # 4) Фильтруем под БД
    df_filtered = filter_for_db(df_clean)
    df_filtered.to_csv("books_filtered.csv", index=False, encoding="utf-8-sig")
    print(f"[csv] saved filtered: {len(df_filtered)} rows -> books_filtered.csv")

    # 5) Загрузка в Postgres (по IP в docker bridge)
    pg_dsn = "postgresql+psycopg2://myuser:mypassword@172.17.0.3:5432/mydatabase"
    load_to_postgres(df_filtered, pg_dsn, table_name="books_filtered")

    # 6) Быстрая проверка, что данные реально в БД
    engine = create_engine(pg_dsn)
    check = pd.read_sql("SELECT COUNT(*) AS cnt FROM books_filtered", engine)
    print("[db] rows in books_filtered:", int(check.loc[0, "cnt"]))


if __name__ == "__main__":
    main()