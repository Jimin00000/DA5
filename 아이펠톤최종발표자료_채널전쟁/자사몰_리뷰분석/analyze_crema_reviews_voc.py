# ================================================================
# [자사몰(고도몰) + 스마트스토어 통합] 소구점/감성 분석
#
# 사전 조건: crawl_crema_reviews()로 받은 df (또는 csv를 다시 읽은 df)에
#            review_id, score, channel, review_text, created_date,
#            product_name 컬럼이 있어야 합니다.
#
# 분류 로직: 쿠팡/CJ 분석에서 쓰던 appeal_keywords + 문장 단위 감성분석 그대로 재사용
#            ("키" 키워드 정밀도 보정 버전 포함)
# ================================================================
import re
import pandas as pd

# ── 1. 소구점 키워드 (보정된 버전 — "키" substring 충돌 수정) ────
appeal_keywords = {
    "효능/효과": ["효과", "성장", "면역", "건강", "도움", "변화", "체감", "흡수", "흡수율",
                "염증", "피로", "컨디션", "붓기", "관절"],   # "키" 키워드 제거됨
    "맛/복용편의": ["맛", "비린", "비린내", "거부감", "냄새", "젤리", "구미", "식감", "쓴맛",
                "삼키", "씹", "캡슐", "알약", "향"],
    "성분/안전성": ["성분", "함량", "원료", "인증", "무첨가", "안전", "부작용", "HACCP", "GMP"],
    "가격/가치": ["가격", "가성비", "비싸", "저렴", "대용량", "재구매", "용량"],
    "배송/포장/품질": ["배송", "포장", "파손", "유통기한", "품질", "이물질", "부서", "박스"],
}

# "키(신장)" 전용 정규식 - "삼키다" 등 substring 오염 방지
HEIGHT_PATTERN = re.compile(r'(?<![가-힣])키(?:가|는|를|만|도)?(?=\s|[.,!?]|$)')


def tag_text(text: str, pattern_map: dict) -> list:
    tags = [tag for tag, pats in pattern_map.items() if any(p in text for p in pats)]
    if "효능/효과" not in tags and HEIGHT_PATTERN.search(text):
        tags.append("효능/효과")
    return tags


# ── 2. 긍정/부정 단어 사전 (쿠팡/CJ 분석과 동일) ──────────────────
positive_words = [
    "만족", "좋", "추천", "효과적", "도움", "개선", "재구매", "잘 먹", "잘먹",
    "편하", "신뢰", "기대", "장점", "깔끔", "넉넉", "적당", "괜찮", "잘 섭취",
    "거부감 없", "거부감 적", "흡수율이 높", "맛있", "간식처럼",
]
negative_words = [
    "불만", "실망", "아쉬", "거부", "부작용", "별로", "못 먹", "안 먹",
    "쓴맛", "비린", "불편", "환불", "반품", "후회", "문제", "파손", "부서",
    "짧", "낮", "걸렸", "포기", "거부했", "거부한", "토할", "뱉", "텁텁",
    "맛이 없", "맛없", "기대에 못 미", "재구매 의사가 없", "재구매 의사는 낮",
]


def split_sentences(text: str) -> list:
    text = str(text)
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]


import re as _re

# "비린/거부감/냄새/쓴맛" 등 부정 단어가 부정어(안/없/못 등)와 같은 문장 내
# 가까운 거리에서 같이 나오면 의미가 반전됨 (예: "비린내 안 나요", "거부감 없이")
# 이 패턴에 걸리면 negative_words 매칭 점수에서 빼고 positive 쪽으로 가산 처리
NEGATION_FLIP_PATTERN = _re.compile(
    r'(비린|냄새|거부감|쓴맛|불편)[^.!?]{0,10}(안\s?나|없|않|못\s)|'
    r'(안\s?나|없|않|못\s)[^.!?]{0,10}(비린|냄새|거부감|쓴맛|불편)|'
    r'(비린|냄새)\S{0,4}(안\s?나)'
)
# 한계: "A는 비린내 나는데 B는 안나요"처럼 한 문장에 두 대상이 비교되는 경우는
# 거리가 멀어 못 잡을 수 있음. 문장 분리 단위가 짧을수록(마침표/줄바꿈 기준) 정확도가 높아짐.


def sentence_sentiment(sentence: str) -> str:
    # 부정어로 반전된 단어는 negative 카운트에서 빼고 다시 셈
    neg_flip_matches = NEGATION_FLIP_PATTERN.findall(sentence)
    flipped_count = len(neg_flip_matches)

    pos = sum(sentence.count(w) for w in positive_words)
    neg = sum(sentence.count(w) for w in negative_words)
    neg = max(0, neg - flipped_count)  # 반전된 만큼 부정 점수 차감
    pos += flipped_count               # 반전된 만큼 긍정 쪽으로 가산 (없어서 좋다는 의미)

    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ── 3. 리뷰 단위 분석 (제품/채널 라벨을 같이 들고 다님) ───────────
def analyze_reviews_with_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: review_id, channel, review_text, score, product_name 컬럼 필요
    반환: (review_id, channel, product_name, score, sentence, 카테고리, sentiment) 단위로 explode된 long-format df
    """
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("review_text", ""))
        for sent in split_sentences(text):
            cats = tag_text(sent, appeal_keywords)
            if not cats:
                continue
            sentiment = sentence_sentiment(sent)
            for cat in cats:
                rows.append({
                    "review_id": r.get("review_id"),
                    "channel": r.get("channel"),
                    "product_name": r.get("product_name"),
                    "score": r.get("score"),
                    "sentence": sent,
                    "category": cat,
                    "sentiment": sentiment,
                })
    return pd.DataFrame(rows)


def build_summary_tables(long_df: pd.DataFrame):
    """
    long_df: analyze_reviews_with_meta() 결과
    반환: (통합표, 채널별표, 채널별비율표)
    """
    # ── 통합(전체) 카테고리 x 감성 ──
    combined = (
        long_df.groupby(["category", "sentiment"])
        .size().unstack(fill_value=0)
        .reindex(columns=["positive", "negative", "neutral"], fill_value=0)
        .reindex(appeal_keywords.keys())
        .fillna(0).astype(int)
    )
    combined.columns = ["긍정", "부정", "중립"]
    combined.index.name = "카테고리"

    # ── 채널별 카테고리 x 감성 (절대 건수) ──
    by_channel = (
        long_df.groupby(["channel", "category", "sentiment"])
        .size().unstack(fill_value=0)
        .reindex(columns=["positive", "negative", "neutral"], fill_value=0)
    )
    by_channel.columns = ["긍정", "부정", "중립"]

    # ── 채널별 비율(%) — 채널 간 표본 크기 차이를 보정해서 비교하기 위함 ──
    channel_totals = long_df.groupby("channel").size()
    by_channel_pct = by_channel.div(
        by_channel.groupby(level="channel").transform("sum").replace(0, pd.NA)
    ).fillna(0) * 100
    by_channel_pct = by_channel_pct.round(1)

    return combined, by_channel, by_channel_pct


def build_product_summary(long_df: pd.DataFrame, top_n_products: int = 10) -> pd.DataFrame:
    """제품별 상위 소구점 요약 (리뷰 건수 많은 상품 top_n개만)"""
    top_products = (
        long_df.drop_duplicates("review_id")["product_name"]
        .value_counts().head(top_n_products).index
    )
    sub = long_df[long_df["product_name"].isin(top_products)]
    pivot = (
        sub.groupby(["product_name", "category"]).size()
        .unstack(fill_value=0)
        .reindex(columns=appeal_keywords.keys(), fill_value=0)
    )
    pivot["리뷰수"] = sub.drop_duplicates("review_id").groupby("product_name").size()
    return pivot.sort_values("리뷰수", ascending=False)


if __name__ == "__main__":
    # 사용 예시
    # df = pd.read_csv("phytonutri_crema_reviews_2025/crema_reviews_full.csv")
    # long_df = analyze_reviews_with_meta(df)
    # combined, by_channel, by_channel_pct = build_summary_tables(long_df)
    # product_summary = build_product_summary(long_df)
    pass

# ================================================================
# 예쁜 표 렌더링 (이전 쿠팡/CJ 분석에서 쓰던 스타일 재사용)
# 노트북에서: display_voc_tables(combined, by_channel, by_channel_pct)
# ================================================================
from IPython.display import display


def display_voc_tables(combined: pd.DataFrame, by_channel: pd.DataFrame, by_channel_pct: pd.DataFrame):
    def style_common(df, caption, fmt_pct=False):
        # 행마다 줄무늬(흰색/아주 옅은 회색)를 명시적으로 깔아서
        # 다크모드 노트북에서도 글씨가 항상 보이도록 함
        def zebra_rows(row):
            idx = df.index.get_loc(row.name)
            bg = "#FFFFFF" if idx % 2 == 0 else "#F2F2F2"
            return [f"background-color: {bg}; color: #111111;"] * len(row)

        styled = (
            df.style
            .apply(zebra_rows, axis=1)
            .set_properties(**{
                "font-size": "12px", "font-family": "Arial, sans-serif",
                "text-align": "center",
                "border": "1px solid #d0d0d0", "padding": "6px 10px",
            })
            .set_table_styles([
                {"selector": "thead th", "props": [
                    ("background-color", "#2F5496"), ("color", "white"),
                    ("font-weight", "bold"), ("font-size", "11px"),
                    ("text-align", "center"), ("padding", "7px"),
                ]},
                {"selector": "th.row_heading", "props": [
                    ("background-color", "#E8EEF7"), ("color", "#1F3864"),
                    ("text-align", "left"), ("font-weight", "bold"),
                    ("font-size", "11px"), ("padding", "6px 10px"),
                ]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
                {"selector": "caption", "props": [
                    ("caption-side", "top"), ("font-size", "14px"), ("font-weight", "bold"),
                    ("color", "#1F3864"), ("padding", "10px 0 6px 0"), ("text-align", "left"),
                ]},
            ])
            .set_caption(caption)
        )
        if fmt_pct:
            styled = styled.format("{:.1f}%")
        return styled

    display(style_common(combined, "통합(자사몰+스마트스토어) 소구점 카테고리별 감성 분포"))
    display(style_common(by_channel, "채널별 소구점 카테고리별 감성 분포 (절대 건수)"))
    display(style_common(by_channel_pct, "채널별 소구점 카테고리별 감성 비율 (%, 채널 내부 100% 기준)", fmt_pct=True))
