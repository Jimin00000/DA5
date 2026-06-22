# ================================================================
# 🎁가정의 달 시크릿 링크🎁 89건 - 실제 product_code 확인용
#
# 사전 조건: df (crema_reviews_full.csv 원본)
# ================================================================
import pandas as pd


def inspect_promo_rows(df: pd.DataFrame, keyword: str = "시크릿 링크") -> pd.DataFrame:
    """
    프로모션 페이지로 등록된 리뷰들의 product_code 분포와
    실제 리뷰 내용을 함께 보여줌. 어떤 진짜 제품 얘기인지 파악하는 용도.
    """
    promo = df[df["product_name"].str.contains(keyword, na=False)].copy()
    print(f"=== '{keyword}' 포함 리뷰: 총 {len(promo)}건 ===\n")

    print("--- product_code별 건수 ---")
    code_counts = promo["product_code"].value_counts()
    print(code_counts)

    print("\n--- product_url별 건수 (코드와 1:1 대응 확인용) ---")
    print(promo["product_url"].value_counts())

    return promo


def show_promo_samples(promo: pd.DataFrame, n_per_code: int = 3) -> None:
    """각 product_code별로 실제 리뷰 텍스트 샘플을 보여줌"""
    pd.set_option("display.max_colwidth", 150)
    for code, group in promo.groupby("product_code"):
        print(f"\n=== product_code={code} (해당 리뷰 {len(group)}건) ===")
        print(group[["review_text", "score", "created_date"]].head(n_per_code).to_string(index=False))


if __name__ == "__main__":
    pass
