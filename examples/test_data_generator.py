"""テストデータ生成スクリプト。

Azure AI Search 無料枠でテスト可能なサンプルデータセットを生成する。
商品カタログ（ECサイト）を模したデータで、以下のフィールド型を網羅する:
- Edm.String（キー、テキスト）
- Edm.Int32（整数）
- Edm.Double（浮動小数点）
- Edm.Boolean（真偽値）
- Edm.DateTimeOffset（日時）
- Edm.GeographyPoint（地理座標）
- Collection(Edm.String)（文字列配列）
- Collection(Edm.Single)（ベクトル = FLOAT_VECTOR）

無料枠の制約:
- ストレージ上限: 50MB
- インデックス数上限: 3
- ベクトル次元: 128（ストレージ節約のため）
- ドキュメント数: 200件

使い方:
    python examples/test_data_generator.py [--output test_data.json] [--count 200]
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone

# 再現可能な結果のためにシードを固定
random.seed(42)

# --- 定数 ---
VECTOR_DIM = 128
DEFAULT_DOC_COUNT = 200

# サンプルデータのバリエーション
CATEGORIES = ["エレクトロニクス", "書籍", "衣料品", "食品", "スポーツ", "家具", "玩具", "文房具"]
BRANDS = ["TechCo", "BookHouse", "StyleWear", "GourmetLife", "ActiveGear", "HomeComfort", "FunFactory", "PenCraft"]
COLORS = ["赤", "青", "緑", "黒", "白", "黄", "紫", "橙"]
MATERIALS = ["プラスチック", "金属", "木材", "布", "ガラス", "レザー", "シリコン", "紙"]
TAGS_POOL = [
    "セール", "新着", "人気", "限定", "おすすめ", "送料無料",
    "ギフト対応", "エコ", "国産", "輸入品", "訳あり", "予約",
    "レビュー高評価", "初回限定", "数量限定", "季節限定",
]

# 日本の主要都市の緯度経度（GeographyPoint テスト用）
JP_CITIES = [
    {"name": "東京", "lat": 35.6762, "lon": 139.6503},
    {"name": "大阪", "lat": 34.6937, "lon": 135.5023},
    {"name": "名古屋", "lat": 35.1815, "lon": 136.9066},
    {"name": "札幌", "lat": 43.0618, "lon": 141.3545},
    {"name": "福岡", "lat": 33.5904, "lon": 130.4017},
    {"name": "仙台", "lat": 38.2682, "lon": 140.8694},
    {"name": "広島", "lat": 34.3853, "lon": 132.4553},
    {"name": "横浜", "lat": 35.4437, "lon": 139.6380},
    {"name": "京都", "lat": 35.0116, "lon": 135.7681},
    {"name": "神戸", "lat": 34.6901, "lon": 135.1956},
]

# 商品名テンプレート
PRODUCT_TEMPLATES = [
    "{brand} {material}製 {color} {category}アイテム",
    "{brand} プレミアム {color} {category}",
    "{brand} {material} {category} スペシャル",
    "{color} {material} {category} by {brand}",
    "{brand} {category} デラックス {color}",
]


def generate_vector(dim: int) -> list[float]:
    """ランダムな正規化ベクトルを生成する。"""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(v * v for v in vec) ** 0.5
    return [round(v / norm, 6) for v in vec]


def generate_document(idx: int) -> dict:
    """1件の商品ドキュメントを生成する。"""
    category = random.choice(CATEGORIES)
    brand = random.choice(BRANDS)
    color = random.choice(COLORS)
    material = random.choice(MATERIALS)
    city = random.choice(JP_CITIES)

    # 商品名を生成
    template = random.choice(PRODUCT_TEMPLATES)
    title = template.format(
        brand=brand, material=material, color=color, category=category
    )

    # タグをランダムに選択（1〜5個）
    tags = random.sample(TAGS_POOL, k=random.randint(1, 5))

    # 日時を過去1年間のランダムな日時にする
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    random_days = random.randint(0, 365)
    random_hours = random.randint(0, 23)
    created_at = base_date + timedelta(days=random_days, hours=random_hours)

    # 説明文を生成
    description = (
        f"{brand}の{category}カテゴリーの商品です。"
        f"{material}製で{color}色のデザインが特徴。"
        f"{city['name']}の倉庫から発送されます。"
    )

    return {
        "id": f"product-{idx:04d}",
        "title": title,
        "description": description,
        "category": category,
        "brand": brand,
        "price": round(random.uniform(100, 50000), 2),
        "rating": random.randint(1, 5),
        "in_stock": random.choice([True, False]),
        "created_at": created_at.isoformat(),
        "tags": tags,
        "warehouse_location": {
            "type": "Point",
            "coordinates": [city["lon"], city["lat"]],
        },
        "content_vector": generate_vector(VECTOR_DIM),
    }


def generate_dataset(count: int = DEFAULT_DOC_COUNT) -> list[dict]:
    """指定件数の商品ドキュメントを生成する。"""
    return [generate_document(i) for i in range(count)]


def estimate_size_mb(docs: list[dict]) -> float:
    """生成データのおおよそのサイズ（MB）を推定する。"""
    raw = json.dumps(docs, ensure_ascii=False)
    return len(raw.encode("utf-8")) / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Azure AI Search テスト用データセット生成"
    )
    parser.add_argument(
        "--output", "-o",
        default="test_data.json",
        help="出力ファイルパス（デフォルト: test_data.json）",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=DEFAULT_DOC_COUNT,
        help=f"生成するドキュメント数（デフォルト: {DEFAULT_DOC_COUNT}）",
    )
    args = parser.parse_args()

    print(f"テストデータ生成中... ({args.count} 件)")
    docs = generate_dataset(args.count)

    size_mb = estimate_size_mb(docs)
    print(f"推定データサイズ: {size_mb:.2f} MB")

    if size_mb > 45:
        print("⚠ 警告: 無料枠の 50MB 制限に近いです。--count を減らしてください。")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"生成完了: {args.output} ({args.count} 件, {size_mb:.2f} MB)")

    # フィールド型の一覧を表示
    print("\n--- フィールド構成 ---")
    field_types = {
        "id": "Edm.String (Key)",
        "title": "Edm.String (Searchable)",
        "description": "Edm.String (Searchable)",
        "category": "Edm.String (Filterable, Facetable)",
        "brand": "Edm.String (Filterable, Facetable)",
        "price": "Edm.Double (Filterable, Sortable)",
        "rating": "Edm.Int32 (Filterable, Sortable, Facetable)",
        "in_stock": "Edm.Boolean (Filterable)",
        "created_at": "Edm.DateTimeOffset (Filterable, Sortable)",
        "tags": "Collection(Edm.String) (Filterable, Facetable)",
        "warehouse_location": "Edm.GeographyPoint (Filterable)",
        "content_vector": f"Collection(Edm.Single) ({VECTOR_DIM}次元, HNSW/cosine)",
    }
    for name, type_desc in field_types.items():
        print(f"  {name}: {type_desc}")


if __name__ == "__main__":
    main()
