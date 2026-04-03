import pandas as pd
import re
import ast
from config import DATA_PATH, TEXT_COLUMNS, JOB_COLUMNS, JOB_LABELS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s+#.\-/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_labels(label_str):
    if isinstance(label_str, list):
        return label_str
    try:
        return ast.literal_eval(str(label_str))
    except Exception:
        return []


def load_data(path=DATA_PATH):
    print("Loading data from: " + path)
    df = pd.read_csv(path)
    print("   Rows: " + str(len(df)))

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('')

    df['combined_text'] = (
        df['Text'] + " " + df['Skills'] + " " +
        df['Education'] + " " + df['Experience'] + " " +
        df['Additional_Information']
    )
    df['clean_text'] = df['combined_text'].apply(clean_text)

    if 'Labels' in df.columns:
        df['Labels_parsed'] = df['Labels'].apply(parse_labels)
        df['Job_Names'] = df['Labels_parsed'].apply(
            lambda labels: [JOB_LABELS.get(l, 'Unknown') for l in labels]
        )

    documents = []
    for idx, row in df.iterrows():
        jobs = [j for j in JOB_COLUMNS if row.get(j, 0) == 1]
        doc = {
            'id': str(idx),
            'text': row['clean_text'],
            'skills': str(row.get('Skills', '')),
            'education': str(row.get('Education', '')),
            'experience': str(row.get('Experience', '')),
            'additional': str(row.get('Additional_Information', '')),
            'jobs': jobs,
            'jobs_str': ', '.join(jobs),
        }
        documents.append(doc)

    print("   Documents: " + str(len(documents)))
    return df, documents


def get_job_documents(df, documents):
    job_docs = []
    for job_name in JOB_COLUMNS:
        if job_name not in df.columns:
            continue
        mask = df[job_name] == 1
        job_df = df[mask]
        if len(job_df) == 0:
            continue
        all_skills = " ".join(job_df['Skills'].fillna('').tolist())[:2000]
        all_education = " ".join(job_df['Education'].fillna('').tolist())[:1000]
        all_experience = " ".join(job_df['Experience'].fillna('').tolist())[:2000]
        job_text = (
            "Job Role: " + job_name.replace('_', ' ') + ". "
            "Common Skills: " + all_skills[:500] + ". "
            "Typical Education: " + all_education[:300] + ". "
            "Typical Experience: " + all_experience[:500] + "."
        )
        job_docs.append({
            'id': "job_" + job_name,
            'text': clean_text(job_text),
            'job_name': job_name,
            'count': int(mask.sum()),
            'type': 'job_profile'
        })
    print("   Job profiles: " + str(len(job_docs)))
    return job_docs