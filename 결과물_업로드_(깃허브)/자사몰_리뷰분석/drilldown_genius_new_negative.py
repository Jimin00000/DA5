# ================================================================
# 지니어스뉴(드롭스&톡캡스) 맛/복용편의 부정 리뷰 drill-down
#
# 사전 조건: df (crema_reviews_full.csv 읽은 원본),
#            analyze_reviews_with_meta, tag_text, split_sentences,
#            sentence_sentiment 가 이미 로드되어 있어야 합니다.
#            (analyze_crema_reviews_voc.py를 먼저 import/실행했다면 OK)
# ================================================================
import pandas as pd
from collections import Counter
from analyze_crema_reviews_voc import appeal_keywords, tag_text, split_sentences, sentence_sentiment

TARGET_PRODUCT = "지니어스뉴 드롭스&톡캡스"


def drilldown_negative(df: pd.DataFrame, product_name: str = TARGET_PRODUCT,
                        category: str = "맛/복용편의") -> pd.DataFrame:
    """
    특정 제품 x 특정 카테고리의 '부정' 문장만 모아서,
    원본 리뷰 메타(채널/평점/작성일)와 함께 보여줌.
    """
    rows = []
    sub = df[df["product_name"] == product_name]
    for _, r in sub.iterrows():
        text = str(r.get("review_text", ""))
        for sent in split_sentences(text):
            cats = tag_text(sent, appeal_keywords)
            if category not in cats:
                continue
            sentiment = sentence_sentiment(sent)
            if sentiment != "negative":
                continue
            rows.append({
                "review_id": r.get("review_id"),
                "channel": r.get("channel"),
                "score": r.get("score"),
                "created_date": r.get("created_date"),
                "sentence": sent,
                "full_review_text": text,
            })
    return pd.DataFrame(rows)


def keyword_frequency_in_negative(neg_df: pd.DataFrame, top_k: int = 20) -> pd.Series:
    """부정 문장들 안에서 어떤 맛/복용편의 키워드가 가장 많이 나오는지"""
    keywords = appeal_keywords["맛/복용편의"]
    counter = Counter()
    for sent in neg_df["sentence"]:
        for kw in keywords:
            if kw in sent:
                counter[kw] += 1
    return pd.Series(counter).sort_values(ascending=False).head(top_k)


if __name__ == "__main__":
    pass
