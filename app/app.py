from pathlib import Path
import base64

import gradio as gr
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

MOVIES_PATH = BASE_DIR / "data" / "raw" / "movies.parquet"
RATINGS_PATH = BASE_DIR / "data" / "processed" / "ready_to_train_1M.parquet"

movies_df = pd.read_parquet(MOVIES_PATH).copy()
ratings_df = pd.read_parquet(RATINGS_PATH).copy()
high_ratings_df = ratings_df[
    ratings_df["target_rating"] >= 4
][["userId", "movieId", "target_rating"]].copy()

movies_df["movieYear"] = pd.to_numeric(movies_df["movieYear"], errors="coerce").fillna(0)
movies_df["audience_score"] = pd.to_numeric(movies_df["audience_score"], errors="coerce")

movies_df["display_title"] = (
    movies_df["movieTitle"].astype(str)
    + " ("
    + movies_df["movieYear"].astype(int).astype(str)
    + ")"
)
movies_df = movies_df.drop_duplicates(
    subset=["display_title"]
).reset_index(drop=True)

movie_title_to_id = dict(zip(movies_df["display_title"], movies_df["movieId"]))
movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["display_title"]))
movie_info = (
    movies_df
    .drop_duplicates(subset=["display_title"])
    .set_index("display_title")
    .to_dict("index")
)

valid_movie_ids = set(ratings_df["movieId"].unique())
movie_choices = [
    title for title, movie_id in movie_title_to_id.items()
    if movie_id in valid_movie_ids
]

STAR_CHOICES = [
    ("⭐", 1),
    ("⭐⭐", 2),
    ("⭐⭐⭐", 3),
    ("⭐⭐⭐⭐", 4),
    ("⭐⭐⭐⭐⭐", 5),
]


def encode_background() -> str:
    for name in ["bg_image.jpg", "bg_image.jpeg", "bg_image.png"]:
        path = APP_DIR / "assets" / name
        if path.exists():
            with open(path, "rb") as img:
                return base64.b64encode(img.read()).decode()
    return ""


encoded_bg = encode_background()

bg_css = (
    f"""
    background:
        linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.92)),
        url("data:image/jpeg;base64,{encoded_bg}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    """
    if encoded_bg
    else
    """
    background: linear-gradient(135deg, #050505, #141414, #2b1055) !important;
    """
)

CUSTOM_CSS = f"""
.gradio-container {{
    {bg_css}
    min-height: 100vh !important;
    color: white !important;
    font-family: Inter, Segoe UI, sans-serif !important;
}}

#main-title {{
    text-align: center;
    font-size: 64px;
    font-weight: 950;
    letter-spacing: 1px;
    color: white;
}}

#main-title span {{
    color: #e50914;
}}

#subtitle {{
    text-align: center;
    font-size: 20px;
    color: #d8d8d8;
    margin-bottom: 28px;
}}

#search-panel, #selected-panel, #recommend-panel {{
    background: rgba(15,15,15,0.86);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.55);
}}

.movie-card {{
    min-height: 150px;
    background: linear-gradient(145deg, rgba(35,35,35,.96), rgba(10,10,10,.96));
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 30px rgba(0,0,0,0.55);
}}

.movie-title {{
    font-size: 20px;
    font-weight: 850;
    color: white;
}}

.movie-meta {{
    margin-top: 8px;
    font-size: 15px;
    color: #cfcfcf;
}}

.score-pill {{
    margin-top: 14px;
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(229,9,20,0.18);
    color: #ffcece;
    font-weight: 700;
}}

.empty-card {{
    min-height: 150px;
    border: 1px dashed rgba(255,255,255,0.25);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #aaa;
    background: rgba(255,255,255,0.04);
}}

#recommend-btn {{
    background: #e50914 !important;
    color: white !important;
    font-weight: 900 !important;
    border-radius: 14px !important;
    padding: 14px !important;
}}
"""


def movie_card_html(movie_title: str) -> str:
    if not movie_title:
        return "<div class='empty-card'>Add Movie</div>"

    info = movie_info.get(movie_title, {})
    year = int(info.get("movieYear", 0))
    score = info.get("audience_score", None)
    score_text = "N/A" if pd.isna(score) else f"{int(score)}%"

    title = info.get("movieTitle", movie_title)

    return f"""
    <div class="movie-card">
        <div class="movie-title">{title}</div>
        <div class="movie-meta">Year: {year}</div>
        <div class="score-pill">🍿 Audience Score: {score_text}</div>
    </div>
    """


def recommendation_card_html(movie_id: str) -> str:
    title = movie_id_to_title.get(movie_id, "Unknown Movie")
    info = movie_info.get(title, {})
    year = int(info.get("movieYear", 0))
    score = info.get("audience_score", None)
    score_text = "N/A" if pd.isna(score) else f"{int(score)}%"

    movie_title = info.get("movieTitle", title)

    return f"""
    <div class="movie-card">
        <div class="movie-title">{movie_title}</div>
        <div class="movie-meta">Year: {year}</div>
        <div class="score-pill">🍿 Audience Score: {score_text}</div>
    </div>
    """


def add_movie(movie, selected_movies):
    selected_movies = selected_movies or []

    if movie and movie not in selected_movies and len(selected_movies) < 5:
        selected_movies.append(movie)

    padded = selected_movies + [""] * (5 - len(selected_movies))

    return (
        selected_movies,
        movie_card_html(padded[0]),
        movie_card_html(padded[1]),
        movie_card_html(padded[2]),
        movie_card_html(padded[3]),
        movie_card_html(padded[4]),
    )


def clear_movies():
    empty_card = movie_card_html("")

    return (
        [],                    # selected_state

        empty_card,            # card1
        5,                     # rating1

        empty_card,            # card2
        5,                     # rating2

        empty_card,            # card3
        5,                     # rating3

        empty_card,            # card4
        5,                     # rating4

        empty_card,            # card5
        5,                     # rating5

        None,                  # movie_search
        "",                    # recommendations output
    )

def recommend_movies(selected_movies, rating1, rating2, rating3, rating4, rating5):
    selected_movies = selected_movies or []
    ratings = [rating1, rating2, rating3, rating4, rating5]

    if not selected_movies:
        return "Please add and rate at least one movie."

    movie_weights = {}

    for movie_title, rating in zip(selected_movies, ratings):
        if not movie_title:
            continue

        rating = float(rating)

        if rating >= 3:
            movie_id = movie_title_to_id[movie_title]
            movie_weights[movie_id] = rating / 5.0

    liked_movie_ids = list(movie_weights.keys())

    if not liked_movie_ids:
        return "Please rate at least one selected movie 3 stars or higher."

    user_overlap = (
        high_ratings_df[
            high_ratings_df["movieId"].isin(liked_movie_ids)
        ]
        .groupby("userId")
        .size()
    )

    similar_users = user_overlap[user_overlap >= 2].index

    if len(similar_users) == 0:
        similar_users = user_overlap[user_overlap >= 1].index

    if len(similar_users) == 0:
        return "No similar users found. Try different movie choices."

    candidate_ratings = high_ratings_df[
        (high_ratings_df["userId"].isin(similar_users))
        & (~high_ratings_df["movieId"].isin(liked_movie_ids))
    ]

    if candidate_ratings.empty:
        return "No recommendations found. Try different movie choices."

    ranked = (
        candidate_ratings
        .groupby("movieId")
        .agg(
            avg_rating=("target_rating", "mean"),
            rating_count=("target_rating", "count"),
        )
        .reset_index()
    )
    ranked = ranked.merge(
        movies_df[
            [
                "movieId",
                "audience_score",
                "critic_score",
                "movieYear",
            ]
        ],
        on="movieId",
        how="left",
    )

    ranked["audience_score"] = pd.to_numeric(
        ranked["audience_score"], errors="coerce"
    ).fillna(50)

    ranked["critic_score"] = pd.to_numeric(
        ranked["critic_score"], errors="coerce"
    ).fillna(50)

    ranked["movieYear"] = pd.to_numeric(
        ranked["movieYear"], errors="coerce"
    ).fillna(0)

    ranked = ranked[
        (ranked["audience_score"] >= 60)
        & (ranked["movieYear"] >= 1980)
    ]

    ranked["score"] = (
        ranked["avg_rating"]
        * np.log1p(ranked["rating_count"])
        * (ranked["audience_score"] / 100)
        * (ranked["critic_score"] / 100)
    )
    ranked = ranked.sort_values("score", ascending=False)

    cards = []

    for _, row in ranked.iterrows():
        movie_id = row["movieId"]

        if movie_id not in movie_id_to_title:
            continue

        cards.append(recommendation_card_html(movie_id))

        if len(cards) == 10:
            break

    if not cards:
        return "No recommendations found. Try different movie choices."

    return "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:18px;'>" + "".join(cards) + "</div>"


with gr.Blocks() as demo:
    selected_state = gr.State([])

    gr.HTML(
        """
        <div id="main-title">🎬 <span>Cine</span>Match AI</div>
        <div id="subtitle">Discover your next favorite movie.</div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(elem_id="selected-panel"):
                gr.Markdown("##  Pick Movies You Love")

                with gr.Row():
                    with gr.Column():
                        card1 = gr.HTML(movie_card_html(""))
                        rating1 = gr.Radio(STAR_CHOICES, value=5, label="Your Rating", visible=False)
                    with gr.Column():
                        card2 = gr.HTML(movie_card_html(""))
                        rating2 = gr.Radio(STAR_CHOICES, value=5, label="Your Rating", visible=False)
                    with gr.Column():
                        card3 = gr.HTML(movie_card_html(""))
                        rating3 = gr.Radio(STAR_CHOICES, value=5, label="Your Rating", visible=False)
                    with gr.Column():
                        card4 = gr.HTML(movie_card_html(""))
                        rating4 = gr.Radio(STAR_CHOICES, value=5, label="Your Rating", visible=False)
                    with gr.Column():
                        card5 = gr.HTML(movie_card_html(""))
                        rating5 = gr.Radio(STAR_CHOICES, value=5, label="Your Rating", visible=False)

                recommend_btn = gr.Button(" Generate Recommendations", elem_id="recommend-btn")

        with gr.Column(scale=1):
            with gr.Column(elem_id="search-panel"):
                gr.Markdown("## 🔍 Search")
                movie_search = gr.Dropdown(
                    choices=movie_choices,
                    label="Search Movie",
                    value=None,
                    interactive=True,
                    filterable=True,
                    allow_custom_value=False,
                )
                add_button = gr.Button("➕ Add Movie")
                clear_button = gr.Button("🗑️ Clear All")

    with gr.Column(elem_id="recommend-panel"):
        gr.Markdown("##  Recommended For You")
        output = gr.HTML("")

    
    add_button.click(
        fn=add_movie,
        inputs=[movie_search, selected_state],
        outputs=[
            selected_state,
            card1,
            card2,
            card3,
            card4,
            card5,
        ],
        queue=False,
    )

    clear_button.click(
        fn=clear_movies,
        inputs=[],
        outputs=[
            selected_state,

            card1,
            rating1,

            card2,
            rating2,

            card3,
            rating3,

            card4,
            rating4,

            card5,
            rating5,

            movie_search,
            output,
        ],
        queue=False,
    )

    recommend_btn.click(
        fn=recommend_movies,
        inputs=[
            selected_state,
            rating1,
            rating2,
            rating3,
            rating4,
            rating5,
        ],
        outputs=output,
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )