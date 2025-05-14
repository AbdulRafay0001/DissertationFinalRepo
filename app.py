from fastapi import FastAPI, Form, HTTPException
import math, json

app = FastAPI()

# Helper: classical cosine similarity
def cosine(u, v):
    dot = sum(a*b for a, b in zip(u, v))
    na  = math.sqrt(sum(a*a for a in u))
    nb  = math.sqrt(sum(b*b for b in v))
    return dot/(na*nb) if na and nb else 0.0

@app.post("/recommend") 
def recommend(
    attributes: str = Form(...),
    attribute_vectors: str = Form(...),
    user_vector: str = Form(...)
):
    # parse inputs
    try:
        vecs = json.loads(attribute_vectors)
        user = json.loads(user_vector)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in payload: {e}")
    try:
        attrs = json.loads(attributes)
    except json.JSONDecodeError:
        attrs = attributes.strip().split()

    # compute scores
    scores = []
    for item, vec_dict in vecs.items():
        try:
            v = [vec_dict[a] for a in attrs]
        except KeyError as e:
            raise HTTPException(400, f"Missing attribute {e} in item {item}")
        scores.append((item, cosine(user, v)))

    # top-3
    top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    # log recommendations
    with open("recommendations.log", "a") as log:
        line = ", ".join(f"{item} {score:.3f}" for item, score in top3)
        log.write(line + "\n")

    # return JSON
    return {
        "recommendations": [
            {"item": item, "score": round(score, 3)}
            for item, score in top3
        ]
    }

@app.post("/recommend_tfidf_user")
def recommend_tfidf_user(
    database:     str = Form(...),
    attributes:   str = Form(...),
    user_vector:  str = Form(...)
):
    # strip leading assignment if present
    db_str = database.strip()
    if db_str.startswith(("movies","data")) and "=" in db_str:
        _, _, db_str = db_str.partition("=")
        db_str = db_str.strip()
    try:
        data = json.loads(db_str)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Bad JSON in database: {e}")

    # parse attrs & user
    try:
        attrs = json.loads(attributes)
    except:
        attrs = attributes.strip().split()
    try:
        user = json.loads(user_vector)
    except:
        user = [float(x) for x in user_vector.strip().split()]

    # build TF–IDF matrix
    names = list(data.keys())
    docs  = list(data.values())
    vectorizer   = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    idf_lookup   = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    vocab        = vectorizer.vocabulary_

    # user profile
    profile = [0.0]*len(vocab)
    for attr, score in zip(attrs, user):
        idx = vocab.get(attr.lower())
        if idx is not None:
            profile[idx] = score * idf_lookup[attr.lower()]

    # score items
    sims = []
    for i, name in enumerate(names):
        item_vec = tfidf_matrix[i].toarray()[0].tolist()
        sims.append((name, cosine(profile, item_vec)))

    # return top-3
    top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
    return {"recommendations": [{"item": n, "score": round(s,3)} for n,s in top3]}

@app.post("/recommend_user_refined")
def recommend_user_refined(
    database:          str = Form(...),
    attribute_vectors: str = Form(...),
    attributes:        str = Form(...),
    user_vector:       str = Form(...)
):
    # 1) parse JSON inputs
    try:
        db  = json.loads(database)
        vecs = json.loads(attribute_vectors)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Bad JSON: {e}")

    # 2) attrs & user
    try:
        attrs = json.loads(attributes)
    except:
        attrs = attributes.strip().split()
    try:
        user = json.loads(user_vector)
    except:
        user = [float(x) for x in user_vector.strip().split()]

    # 3) score every item
    def cos(u, v):
        dot = sum(a*b for a,b in zip(u,v))
        nu  = math.sqrt(sum(a*a for a in u))
        nv  = math.sqrt(sum(b*b for b in v))
        return dot/(nu*nv) if nu and nv else 0.0

    scores = []
    for item, vec_dict in vecs.items():
        if all(a in vec_dict for a in attrs):
            v = [vec_dict[a] for a in attrs]
            scores.append((item, cos(user, v)))

    # 4) top-10 → refine
    topN = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
    items      = [itm for itm,_ in topN]
    top_scores = [round(sc,3) for _,sc in topN]
    refined_db = { itm: db.get(itm, "") for itm in items }

    return {
        "top_items":        items,
        "top_scores":       top_scores,
        "refined_database": json.dumps(refined_db)
    }