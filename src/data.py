import re
import pandas as pd
from sklearn.model_selection import train_test_split


class HeadlineDataset:
    """Loads ISOT, cleans headlines, and produces a stratified train/test split."""

    def __init__(self, fake_path, real_path):
        self.fake_path = fake_path
        self.real_path = real_path
        self.df = None

    def load_csv(self):
        fake = pd.read_csv(self.fake_path)
        fake["label"] = "fake"
        real = pd.read_csv(self.real_path)
        real["label"] = "real"
        self.df = pd.concat([fake, real], ignore_index=True)[["title", "subject", "label"]]
        return self

    @staticmethod
    def _clean_title(text):
        t = str(text).lower()
        t = re.sub(r"[\[\(][^\]\)]*[\]\)]", " ", t)                          # [video], (reuters) tags
        t = re.sub(r"https?://\S+", " ", t)                                  # urls
        t = re.sub(r"\b[a-z](?:\.[a-z])+\.?", lambda m: m.group(0).replace(".", ""), t)  # u.s., u.n., e.u. -> us, un, eu
        t = re.sub(r"[’']s\b", "", t)                                        # possessive 's
        t = re.sub(r"[’']", "", t)                                           # other apostrophes: don't -> dont
        t = re.sub(r"\b(video|watch|breaking|factbox)\b", " ", t)            # pure tags
        t = re.sub(r"[^a-z0-9\s]", " ", t)                                   # punctuation
        return re.sub(r"\s+", " ", t).strip()
 
    def clean(self):
        self.df["title"] = self.df["title"].map(self._clean_title)
        self.df = self.df[self.df["title"] != ""]

        # drop titles that appear under both labels (ambiguous), then exact duplicates
        label_counts = self.df.groupby("title")["label"].nunique()
        ambiguous = label_counts[label_counts > 1].index
        self.df = self.df[~self.df["title"].isin(ambiguous)]
        self.df = self.df.drop_duplicates(subset="title").reset_index(drop=True)
        return self

    def split(self, seed=42):
        return train_test_split(
            self.df,
            test_size=0.2,
            stratify=self.df["label"],
            random_state=seed,
        )