import pandas as pd
import requests
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import ALL, State
import numpy as np 
from myfuns import myIBCF
# Fetch movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
response = requests.get(myurl)
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)


smatrix2 = pd.read_csv('https://raw.githubusercontent.com/sudham123/Project4_App/refs/heads/main/output.csv')
movie_mapping = dict(zip(['m' + str(mid) for mid in movies['movie_id']], movies['title']))
first_100_columns = smatrix2.columns
mapped_titles = []
for col in first_100_columns:
    if col in movie_mapping:
        mapped_titles.append(movie_mapping[col])
result_df = pd.DataFrame({
    'movie_id': first_100_columns,
    'title': mapped_titles
})
movies = result_df


def get_displayed_movies():
    return movies.head(100)


def get_movie_card(movie):
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie.movie_id[1:]}.jpg?raw=true",
                    top=True,
                    style={
                        "height": "250px",
                        "width": "200px",
                        "object-fit": "contain",
                        "margin": "auto",
                    },
                ),
                dbc.CardBody(
                    [
                        html.H6(movie.title, className="card-title text-center"),
                        dcc.Slider(
                            min=0,
                            max=5,
                            step=1,
                            value=0,
                            marks={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
                            id={"type": "movie_rating", "movie_id": movie.movie_id},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]
                ),
            ],
            className="h-100",
        ),
        className="col mb-4",
    )

def get_movie_card_fortop10(movie_row):
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie_row['movie_id'][1:]}.jpg?raw=true",
                    top=True,
                    style={
                        "height": "250px",
                        "width": "200px",
                        "object-fit": "contain",
                        "margin": "auto",
                    },
                ),
                dbc.CardBody(
                    [
                        html.H6(movie_row['title'], className="card-title text-center"),
                    ]
                ),
            ],
            className="h-100",
        ),
        className="col mb-4",
    )

def group_movies_per_row():
    movie_cards = [get_movie_card(movie) for index, movie in get_displayed_movies().iterrows()]
    grouped_movies = [movie_cards[i:i + 5] for i in range(0, len(movie_cards), 5)]
    return grouped_movies

def group_movies_for_recommendations(movies_df):
    movie_cards = [get_movie_card_fortop10(row) for index, row in movies_df.iterrows()]
    grouped_movies = [movie_cards[i:i + 5] for i in range(0, len(movie_cards), 5)]
    return grouped_movies




movie_rows = group_movies_per_row()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server  = app.server


def render_rating_movies():
    return dbc.Container(
        [
            html.H4("Rate the movies below:", className="text-center mb-4"),
            html.Div(
                [
                    dbc.Row(row, justify="start") for row in movie_rows
                ],
                style={
                    "height": "80vh",
                    "overflowY": "scroll",
                    "border": "1px solid #ddd",
                    "padding": "10px",
                },
            )
        ],
        fluid=True
    )

app.layout = dbc.Container(
    [
        dcc.Store(id="view_state", data={"view": "ratings"}),  # Store to track state
        html.Div(id="tabs-content", children=render_rating_movies()),  # Default content loaded here
        html.Div(
            [
                dbc.Button(
                    "Get Recommendations",
                    id="get_recommendations",
                    color="primary",
                    className="mb-4",
                    style={"font-size": "20px", "padding": "10px 30px"},
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "center",
                "margin-top": "20px",
            },
        ),
        dbc.Button(
            "Go Back",
            id="go_back",
            color="secondary",
            className="mb-4",
            style={"display": "none"}, 
        ),
        # Modal for warning
        dbc.Modal(
            [
                dbc.ModalHeader("Warning"),
                dbc.ModalBody("You must submit at least one rating before proceeding."),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close_modal",
                        className="ml-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="warning_modal",
            is_open=False,
        ),
    ],
    fluid=True,
)



def map_user_ratings_to_full_vector(user_rated_movies):
    # Create a vector of NaNs indexed by smatrix2's column names
    newuser_vector = pd.Series(np.nan, index=smatrix2.columns)
    
    for title, rating in user_rated_movies.items():
        # Look up the movie in the movies DataFrame
        movie_row = movies[movies['title'] == title]
        if not movie_row.empty:
            #get the corresponding movie_id for that title from movies and set newuser_vector at the movie_id to the rating
            newuser_vector[movie_row.iloc[0]['movie_id']] = rating
    
    return newuser_vector




def map_indices_to_movies(indices, movies_df):
    recommended_movies = movies_df[movies_df['movie_id'].isin(indices)].reset_index()
    print(f"recommnded movies \n {recommended_movies}")
    return recommended_movies





@app.callback(
    [Output("tabs-content", "children"),
     Output("get_recommendations", "style"),
     Output("go_back", "style"),
     Output("warning_modal", "is_open")],
    [Input("get_recommendations", "n_clicks"),
     Input("go_back", "n_clicks"),
     Input("close_modal", "n_clicks")],
    State({"type": "movie_rating", "movie_id": ALL}, "value"),
    prevent_initial_call=False,
)
def handle_navigation(get_click, go_back_click, close_modal_click, all_ratings):
    triggered_id = dash.callback_context.triggered_id or ""

    if triggered_id == "get_recommendations":
        if all(v is None or v == 0 for v in all_ratings):
            return render_rating_movies(), {"display": "block"}, {"display": "none"}, True

        user_ratings = {
            movies.iloc[idx]['title']: int(rating)
            for idx, rating in enumerate(all_ratings) if rating and rating > 0
        }
        print(user_ratings)
        newuser_vector = map_user_ratings_to_full_vector(user_ratings)
        print(f"user chosen {newuser_vector}")
        recommendations = myIBCF(newuser_vector)
        print(f"receomnadtion returned  {recommendations}")
        recommended_movies = map_indices_to_movies(recommendations, movies)
        # print(recommendations)

        recommendation_rows = group_movies_for_recommendations(recommended_movies)

        return dbc.Container(
            [
                html.H4("Top 10 Recommended Movies", className="text-center mb-4"),
                html.Div(
                    [dbc.Row(row, justify="start") for row in recommendation_rows],
                    style={
                        "height": "80vh",
                        "overflowY": "scroll",
                        "border": "1px solid #ddd",
                        "padding": "10px",
                    },
                )
            ],
            fluid=True,
        ), {"display": "none"}, {"display": "block"}, False

    elif triggered_id == "go_back":
        return render_rating_movies(), {"display": "block"}, {"display": "none"}, False

    if triggered_id == "close_modal":
        return render_rating_movies(), {"display": "block"}, {"display": "none"}, False

    return render_rating_movies(), {"display": "block"}, {"display": "none"}, False



if __name__ == "__main__":
    app.run_server(debug=True)

