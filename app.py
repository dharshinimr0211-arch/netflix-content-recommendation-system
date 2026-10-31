import streamlit as st
import pandas as pd
import html

from textwrap import dedent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Netflix Discover",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for application styling
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #071516;
    color: #FFFFFF;
}

.block-container {
    padding-top: 1rem !important;
}

/* Main heading */
h1 {
    color: #448690;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}

h2, h3 {
    color: #FFFFFF;
}

/* Search box */
div[data-baseweb="select"] {
    background-color: #28565D;
    border-radius: 10px;
}

/* Content type buttons */
div[role="radiogroup"] {
    gap: 10px;
}

div[role="radiogroup"] label {
    background-color: #28565D;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid #2B6E7C;
    color: white;
}

div[role="radiogroup"] label:hover {
    background-color: #2B6E7C;
}

/* Selected radio button */
div[role="radiogroup"] label:has(input:checked) {
    background-color: #448690;
    color: white;
    border: 1px solid #448690;
}

/* Recommendation cards */
.movie-card {
    background-color: #10282C;
    padding: 18px;
    border-radius: 12px;
    min-height: 155px;
    margin-bottom: 15px;
    border: 1px solid #28565D;
    overflow-wrap: anywhere;
}

/* Movie title */
.movie-title {
    font-size: 18px;
    font-weight: bold;
    color: #FFFFFF;
    margin-bottom: 12px;
}

/* Movie information */
.movie-info {
    font-size: 14px;
    color: #B8D0D2;
    margin-bottom: 10px;
}

/* Match percentage */
.match-text {
    color: #448690;
    font-size: 14px;
    font-weight: bold;
}

/* Dividers */
hr {
    border: 1px solid #28565D;
}

/* Information messages */
div[data-testid="stAlert"] {
    background-color: #28565D;
    border: 1px solid #448690;
}

</style>
""", unsafe_allow_html=True)


# Load and preprocess the Netflix dataset
@st.cache_data
def load_data():

    df = pd.read_csv("Dataset.csv")

    # Remove duplicate records
    df = df.drop_duplicates(
        subset="show_id"
    ).reset_index(drop=True)

    # Handle missing values in metadata columns
    features = [
        "director",
        "country",
        "rating",
        "listed_in"
    ]

    for col in features:
        df[col] = df[col].fillna("").astype(str)

    df["title"] = df["title"].fillna("").astype(str)
    df["type"] = df["type"].fillna("").astype(str)

    # Combine selected features into one metadata column
    df["metadata"] = (
        df["listed_in"] + " " +
        df["director"] + " " +
        df["country"] + " " +
        df["rating"] + " " +
        df["type"]
    ).str.lower()

    return df

# Convert metadata into TF-IDF vectors
@st.cache_resource
def build_model(metadata):

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(metadata)

    return tfidf_matrix

# Load dataset and build the TF-IDF model
df = load_data()
tfidf_matrix = build_model(df["metadata"])

# Display application title and tagline
st.markdown(
    """
    <h1 style='color:#76190E; text-align:center;'>
    NETFLIX DISCOVER
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center;color:#B8D0D2;'>
        Find more of what you enjoy!
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Allow users to search and select a title
st.subheader("Search for a title")

titles = sorted(
    df[
        df["title"].str.strip() != ""
    ]["title"].unique().tolist()
)

selected_title = st.selectbox(
    "Search",
    titles,
    index=None,
    placeholder="🔍 Search a movie or TV show...",
    label_visibility="collapsed"
)

# Allow users to filter recommendations by content type
st.subheader("Content type")

content_options = [
    "All",
    "Movies",
    "TV Shows"
]

selected_type = st.radio(
    "Content type",
    content_options,
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# Generate recommendations for the selected title
if selected_title:

    # Find the selected title in the dataset
    matching_indices = df.index[
        df["title"] == selected_title
    ].tolist()

    idx = matching_indices[0]

    # Calculate cosine similarity with all titles
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Sort titles based on similarity scores
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude the selected title from recommendations
    similar_indices = [
        i for i in similar_indices
        if i != idx
    ]

    # Create a dataframe containing recommended titles
    recommendations = df.iloc[similar_indices][
        ["title", "type", "listed_in"]
    ].copy()

    recommendations["similarity_score"] = [
        similarity_scores[i]
        for i in similar_indices
    ]

    # Apply the selected content type filter
    if selected_type == "Movies":

        recommendations = recommendations[
            recommendations["type"] == "Movie"
        ]

    elif selected_type == "TV Shows":

        recommendations = recommendations[
            recommendations["type"] == "TV Show"
        ]

    # Keep all filtered recommendations for fallback
    all_recommendations = recommendations.copy()

    # Select titles with similarity scores above 60%
    recommendations = all_recommendations[
        all_recommendations["similarity_score"] > 0.60
    ]

    # Display the three closest matches if no title meets the threshold
    if recommendations.empty:

        recommendations = all_recommendations.head(3)
        fallback_used = True

    else:

        recommendations = recommendations.head(10)
        fallback_used = False

    # Display the recommendation results
    st.subheader(
        "Because you liked " + selected_title
    )

    if recommendations.empty:

        st.warning(
            "No recommendations available for the selected content type."
        )

    else:

        # Display a message based on the recommendation results
        if fallback_used:

            st.info(
                "We couldn't find enough strong matches. "
                "Here are 3 similar titles you might enjoy!"
            )

        else:

            st.write(
                f"Here are top {len(recommendations)} titles picked just for you!"
            )

        # Arrange recommendation cards in two columns
        columns = st.columns(2)

        for i, (_, movie) in enumerate(
            recommendations.iterrows()
        ):

            with columns[i % 2]:

                # Prepare title and metadata for display
                movie_title = html.escape(
                    str(movie["title"])
                )

                movie_type = html.escape(
                    str(movie["type"])
                )

                movie_genre = html.escape(
                    str(movie["listed_in"])
                )

                match_score = float(
                    movie["similarity_score"]
                )

                # Create a card for each recommendation
                card = dedent(f"""
                <div class="movie-card">
                    <div class="movie-title">{movie_title}</div>
                    <div class="movie-info">Type: {movie_type}</div>
                    <div class="movie-info">Genre: {movie_genre}</div>
                    <div class="match-text">
                        Match: {match_score * 100:.1f}%
                    </div>
                </div>
                """)

                st.markdown(
                    card,
                    unsafe_allow_html=True
                )

# Display application footer
st.divider()

st.markdown(
    """
    <p style='text-align:center;color:#448690;font-size:18px;'>
        Search • Discover • Stream
    </p>
    """,
    unsafe_allow_html=True
)