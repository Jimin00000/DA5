# ================================================================
# 자사몰+스마트스토어 리뷰(crema) -> 경쟁사 분석 4개 카테고리 매핑
#
# 기준: 쿠팡/CJ온스타일 경쟁 제품 분석에서 쓴 동일한 3개 카테고리
#   (커큐민+ 는 자사몰 리뷰 데이터에 해당 제품이 없어 비교 대상에서 제외)
#   - 지니어스뉴 (오메가3)
#   - 그로우뉴 (칼마디)
#   - 투데이D3 (비타민D)
#
# 세트 상품(2개 제품 묶음)은 두 카테고리 모두에 중복으로 포함시킴
# (사용자 확인: "두 카테고리에 리뷰를 중복으로 넣기")
# ================================================================
import pandas as pd

CATEGORY_MAP = {
    # ── 지니어스뉴 (오메가3) ──
    "지니어스뉴 드롭스&톡캡스": ["지니어스뉴 (오메가3)"],
    "지니어스뉴 키즈 국내 유일 스퀴즈 짜먹는 어린이 오메가3 DHA ALA 콜린 올로메가": ["지니어스뉴 (오메가3)"],

    # ── 그로우뉴 (칼마디) ──
    "그로우뉴 아이 유기농칼슘&마그네슘 비타민D,k2,망간": ["그로우뉴 (칼마디)"],
    "그로우뉴 ~키즈 유기농 칼슘&마그네슘 비타민D,k2 망간": ["그로우뉴 (칼마디)"],

    # ── 투데이D3 (비타민D) ──
    "투데이디3 for baby": ["투데이D3 (비타민D)"],

    # ── 세트 상품: 두 카테고리에 중복 포함 ──
    "지니어스뉴 + 그로우뉴 세트": ["지니어스뉴 (오메가3)", "그로우뉴 (칼마디)"],
    "데이프로바 + 투데이디3 세트": ["투데이D3 (비타민D)"],  # 데이프로바(유산균)는 4개 카테고리 밖이라 투데이D3만 인정
    "지니어스뉴 + 투데이디3 세트": ["지니어스뉴 (오메가3)", "투데이D3 (비타민D)"],

    # ── 프로모션 페이지 (product_code=1000000402) - 분석 제외 ──
    # [한계점 메모]
    # - 이 product_name(🎁가정의 달 시크릿 링크🎁)에 달린 리뷰는 총 201건
    # - product_meta_reviews_count=22,605로 확인됨: 단일 제품 리뷰수로는
    #   너무 커서, 여러 제품을 한 페이지에서 함께 파는 통합 이벤트 SKU로 추정됨
    #   (지니어스뉴/그로우뉴/투데이D3 등을 한 링크에서 골라 살 수 있는 프로모션 페이지)
    # - 리뷰 텍스트 기반으로 제품을 추론하는 방법을 시도했으나, 실제 리뷰 7건 샘플
    #   검증 결과 6/7건이 "잘 먹어요", "배송이 느렸어요" 같은 일반적 후기로 제품명을
    #   전혀 언급하지 않아 분류 불가능했음 (1건만 "투데이디"가 명시되어 분류됨)
    # - product_options 필드에 구매 옵션(실제 선택 제품)이 있는지 확인했으나
    #   신뢰할 수 있는 형태로 확보되지 않아, 정확한 매핑 근거 부족
    # - 결론: 201건(전체 3,000건의 약 6.7%)은 제품 식별 불가로 판단하여
    #   전체 분석에서 제외함. 보고서/발표 시 이 한계를 명시할 것.
    "🎁가정의 달 맞이 시크릿 링크🎁 오직 이 링크에서만 전제품 할인": [],
    "이트뮨 아이 배도라지즙&엘더베리사과주스": [],
    "데이프로바 for baby 아기 쌩쌩유산균": [],
    "KD 파마 듀얼 초임계 rTG 오메가3": [],
    "뉴스데일리베스트 X 파이토뉴트리 맨드로포즈+": [],
    "수드티 루이보스 허브 블렌드 티 강황 맘먼트 임산부 차": [],
    "리버티엑스": [],
    "듀오좀 마그앤알티지 식물성 듀얼 오메가3": [],
    "옥토피아 X 파이토뉴트리 (PHYTONUTRI)": [],
    "에피베리어 곤약세라미드 1.8": [],
    "이트뮨 엘더베리사과주스": [],
    "인-칼슘 앱솔브+": [],
    "오큘라레이드+": [],
    "요로굿 [소비기한 2026-11-21]": [],
    "곤약세라미드 1.8 에피베리어+ 히알라스킨": [],
    "히알라스킨": [],
    "슬립밸런스": [],
    "블러드슈가케어 헤모웰당": [],
    "유기농 비타민D3 코어": [],  # 이름은 비타민D지만 product_name이 투데이D3와 다름 - 별도 제품으로 보고 보류
    "맨드로포즈+": [],
    "[단종] 조인트리션 릴리버": [],
}

COMPETITOR_CATEGORIES = ["지니어스뉴 (오메가3)", "그로우뉴 (칼마디)", "투데이D3 (비타민D)"]


def classify_promo_by_text(text: str) -> list:
    """
    [폐기된 접근 - 참고용으로만 남겨둠]
    프로모션 페이지 리뷰 텍스트에서 제품 키워드를 찾아 분류를 시도했으나,
    실제 검증 결과 대부분의 리뷰가 제품명을 언급하지 않아(7건 중 6건 분류 불가)
    신뢰할 수 있는 분류 방법이 아님이 확인됨. 사용하지 않음.
    (자세한 내용은 CATEGORY_MAP의 시크릿 링크 항목 주석 참고)
    """
    text = str(text)
    cats = []
    if any(kw in text for kw in ["지니어스뉴", "오메가3", "오메가 3", "DHA", "톡캡스", "드롭스"]):
        cats.append("지니어스뉴 (오메가3)")
    if any(kw in text for kw in ["그로우뉴", "칼슘", "마그네슘", "칼마디"]):
        cats.append("그로우뉴 (칼마디)")
    if any(kw in text for kw in ["투데이디", "투데이D3", "비타민디", "비타민D"]):
        cats.append("투데이D3 (비타민D)")
    return cats


def expand_to_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    product_name 1개 -> 카테고리 0~2개로 매핑하면서 explode.
    세트 상품은 두 행으로 복제됨(중복 포함 요청에 따름).
    매핑 안 된 product_name(프로모션 페이지 포함)은 분석에서 제외됨.
    """
    df = df.copy()
    df["matched_categories"] = df["product_name"].map(
        lambda p: CATEGORY_MAP.get(p, "__UNMAPPED__")
    )

    unmapped = df[df["matched_categories"] == "__UNMAPPED__"]["product_name"].unique()
    if len(unmapped) > 0:
        print(f"⚠️ 매핑표에 없는 product_name {len(unmapped)}개 발견 (이 행들은 분류에서 빠집니다):")
        for p in unmapped:
            print(f"   - {p}")

    # 카테고리 밖으로 명시적으로 분류된 행(빈 리스트, 프로모션 포함)도 여기서 같이 빠짐
    df = df[df["matched_categories"] != "__UNMAPPED__"].copy()
    df_exploded = df.explode("matched_categories")
    df_exploded = df_exploded[df_exploded["matched_categories"].notna()]
    df_exploded = df_exploded.rename(columns={"matched_categories": "competitor_category"})
    return df_exploded[df_exploded["competitor_category"] != ""]


def check_promo_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """프로모션(시크릿 링크) 항목의 실제 product_code 분포 확인용"""
    promo = df[df["product_name"].str.contains("시크릿 링크", na=False)]
    return promo["product_code"].value_counts()


if __name__ == "__main__":
    pass
