import pandas as pd
import re

def auto_label_japanese_politeness(text):
    """Assign politeness level based on recognizable patterns."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Polite: includes honorific verbs or endings
    if re.search(r"(ます|です|ください|ございます|いたします|よろしく|お願いいたします)", text):
        return 2

    # Casual: slang, informal tone, rough pronouns
    elif re.search(r"(お前|だよ|じゃん|かよ|ぜ|よな|なんだよ|そりゃ|やれよ|見て見て|だな)", text):
        return 0

    # Neutral: short interjections, plain forms
    elif re.fullmatch(r"(あ+|え+|うん|ん\?|ふーん|はあ|あれ\?|えーっと.*)", text.strip()):
        return 1

    elif re.search(r"(のか|なんだ|〜てる|〜た|ってこと)", text):
        return 1

    return ""  # unclear, leave blank for manual review

def main():
    input_path = r"politeness-classifier-jp\data\processed\unlabeled_subtitles.csv"
    output_path = r"politeness-classifier-jp\data\processed\semi_labeled_subtitles.csv"

    df = pd.read_csv(input_path)

    df["label"] = df["text"].apply(auto_label_japanese_politeness)

    df.to_csv(output_path, index=False)
    print(f"Labeled data saved to: {output_path}")
    print(df["label"].value_counts(dropna=False))

if __name__ == "__main__":
    main()