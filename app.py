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
