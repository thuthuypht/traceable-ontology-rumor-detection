# Data ethics and public artifact boundaries

**Do not upload the sibling `PRIVATE_RESEARCH_ARCHIVE_DO_NOT_UPLOAD` directory to a public repository.**

Twitter15/16 event data and reply trees originate from the original research releases. Refer to repositories and their own terms:

- Original Twitter15/16: https://github.com/gszswork/Twitter15_16_dataset (pinned tree commit is recorded in `results/twitter_covid19/FROZEN_EXTERNAL_PROTOCOL.json`).
- Twitter15/16 source-text mirror for this study: https://github.com/leekhithnunna/Twitter_15_16_Rumour_Verification (refer to source manifest and pinned commit).
- Twitter-COVID19 archive: https://github.com/DanielLin97/ACLR4RUMOR-NAACL2022 (`data/Twitter/COVID-19/Twitter-COVID19.7z`; checksum and commit in frozen protocol).
- PHEME source: original publicly released rumor dataset, with reuse restrictions to be checked individually by the authors.

This public-ready directory excludes original CSVs containing post text, source tweets, reply messages, encoded source-text Kaggle notebooks, and fitted text pipelines. The *private* archive includes these for the authorized study; possession of them does not automatically grant redistribution rights. Case-level source IDs and RDF provenance are also withheld from the public folder pending legal/ethical review.

The 16 CatBoost models and two fitted TF-IDF/SVD pipelines reside in the private archive because the preprocessors may contain vocabulary learned from third-party text. Model release is a separate author decision after checking platform and source terms.

No labels from Twitter-COVID19 were used to select source-only model thresholds. Source Twitter15/16 and PHEME had repeated development inspection and should not be represented as new unobserved test sets.
