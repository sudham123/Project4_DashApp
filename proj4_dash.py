import pandas as pd
import requests
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import ALL, State
import numpy as np 
# from myfuns import myIBCF
# Fetch movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
response = requests.get(myurl)
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)


# smatrix2 = pd.read_csv('https://raw.githubusercontent.com/sudham123/Project4_App/refs/heads/main/final_smatrix.csv')

smatrix2 = pd.read_csv('simmatrix2_top30.csv')


movie_mapping = dict(zip(['m' + str(mid) for mid in movies['movie_id']], movies['title']))
first_100_columns = smatrix2.columns
print(f"size: {smatrix2.shape}")
mapped_titles = []
for col in first_100_columns:
    if col in movie_mapping:
        mapped_titles.append(movie_mapping[col])
result_df = pd.DataFrame({
    'movie_id': first_100_columns,
    'title': mapped_titles
})
combined_movies = pd.concat([result_df, movies[['movie_id', 'title']]], ignore_index=True)
combined_movies = combined_movies.dropna(subset=['title']).drop_duplicates(subset=['movie_id'])
movies = combined_movies
print(movies)




def anti_join(df_left, df_right, on):
    merged_df = pd.merge(df_left, df_right, on=on, how='left', indicator=True)
    return merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')



def getTopMoviesByRatings():

    avg_ratings = pd.read_csv('test.csv', index_col=0).iloc[:, 0]  # Load as Series
    num_ratings = pd.read_csv('test1.csv', index_col=0)

    # Ensure num_ratings is a Series (extract only the first column if it's a DataFrame)
    num_ratings = num_ratings.iloc[:, 0]

    # Ensure all entries are integers, coerce invalid entries to 0
    num_ratings = pd.to_numeric(num_ratings, errors='coerce').fillna(0).astype(int)

    # Align indices explicitly
    num_ratings = num_ratings.reindex(avg_ratings.index, fill_value=0)  # Ensure alignment with avg_ratings

    # Filter movies with more than 1000 ratings
    movies_filtered = num_ratings > 1000

    # Apply filter to avg_ratings and sort in descending order
    top_10_movies = avg_ratings[movies_filtered].sort_values(ascending=False)

    # Convert index to DataFrame
    result_df = pd.DataFrame(top_10_movies.index)
    print(f"getTopMovies {result_df}")
    return result_df




def myIBCF(smatrix2, newuser):
  previously_rated = np.where(~np.isnan(newuser))[0]
  rated_mov_names = pd.DataFrame(newuser.iloc[previously_rated].index.values)


  unrated = np.where(np.isnan(newuser))[0]
  # print(unrated)
  ratings = newuser.iloc[previously_rated]

  preds = []

  for i in unrated:
    similarity = smatrix2.iloc[i, previously_rated]
    weighted_ratings =  similarity * ratings



    sim_sum = similarity.sum()

    if sim_sum == 0:
      prediction = np.nan
    else:
      prediction = weighted_ratings.sum() / sim_sum
    preds.append(prediction)

  preds_series = pd.Series(preds)

  top_10_predictions = preds_series.nlargest(10)



  # top_10_indices = []
  movie_names = []
  if top_10_predictions.isna().any():

    top_10_predictions = top_10_predictions.dropna()
    filled_amt = len(top_10_predictions)
    top_10_indices = top_10_predictions.index
    un2 = pd.Series(unrated).iloc[top_10_indices]
    movie_names_temp = pd.DataFrame(newuser.iloc[un2].index.values)
    top10_part1 = getTopMoviesByRatings()

    notrated_top = anti_join(top10_part1, rated_mov_names, on=0)
    not_picked = anti_join(notrated_top, movie_names_temp, on=0)

    subs = not_picked.head(10 - filled_amt)
    movie_names = pd.concat([movie_names_temp, subs]).values

  else:
    top_10_indices = top_10_predictions.index
    un2 = pd.Series(unrated).iloc[top_10_indices]
    movie_names = newuser.iloc[un2].index.values




  # top_10_indices = preds_series.nlargest(10).index

  # print(top_10_indices)


  return movie_names



  # map_indicies = unrated.index[top_10_indices]
  # print(top_10_movies)

  # known_ratings = new
  # print(unrated)
  # print(previously_rated)

  # similarity = smatrix2.loc[previously_rated.index]


def get_displayed_movies():
    return movies.head(110)


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
    print(f"Processing movie {movie_row['movie_id'][1:]}: {movie_row['title'] }")
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

# def group_movies_for_recommendations(movies_df):
#     print(f"Total movies in movies_df: {len(movies_df)}")
#     print(f"Movies DataFrame indices: {movies_df.index}")
#     movie_cards = [get_movie_card_fortop10(row) for index, row in movies_df.iterrows()]
#     grouped_movies = [movie_cards[i:i + 5] for i in range(0, len(movie_cards), 5)]
#     # print(f"group {grouped_movies}")
#     return grouped_movies
def get_movie_card_fortop10(movie_row):
    movie_id_str = str(movie_row['movie_id'])
    # print(f"Processing movie {movie_id_str}: {movie_row['title']}")
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie_id_str}.jpg?raw=true",
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


def group_movies_for_recommendations(movies_df):
    # Debugging: Log number of rows and indices in the DataFrame
    # print(f"Total movies in movies_df: {len(movies_df)}")
    # print(f"Movies DataFrame indices: {movies_df.index}")

    # Create movie cards
    movie_cards = []
    for index, row in movies_df.iterrows():
        # print(f"Iterating over index {index}")
        movie_card = get_movie_card_fortop10(row)
        movie_cards.append(movie_card)

    # Debugging: Log the number of movie cards created
    # print(f"Total movie cards created: {len(movie_cards)}")

    # Group movies into sets of 5
    grouped_movies = [movie_cards[i:i + 5] for i in range(0, len(movie_cards), 5)]

    # Debugging: Log the number of groups created
    # print(f"Total groups created: {len(grouped_movies)}")
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




# def map_indices_to_movies(indices, movies_df):
#     # recommended_movies = movies_df[movies_df['movie_id'].isin(indices)].reset_index()
#     # print(f"recommnded movies \n {recommended_movies}")
   
#     return recommended_movies

def flatten_indices(indices):
    """Flatten indices only if they are in a 2D structure."""
    if isinstance(indices, np.ndarray) and indices.ndim > 1: 
        return indices.flatten().tolist()
    elif isinstance(indices[0], list): 
        return [item for sublist in indices for item in sublist]
    return indices 

# def map_indices_to_movies(indices, movies_df):
#     flat_indices = flatten_indices(indices)
#     print("Flattened indices:", flat_indices) 
#     recommended_movies = movies_df[movies_df['movie_id'].isin(flat_indices)].reset_index()
#     return recommended_movies
def map_indices_to_movies(indices, movies_df):
    flat_indices = flatten_indices(indices)
    # print("Flattened indices:", flat_indices)

    # Strip 'm' prefix from the indices
    stripped_indices = [idx[1:] if idx.startswith('m') else idx for idx in flat_indices]
    # print("Stripped indices:", stripped_indices)

    # Perform filtering with the cleaned indices
    recommended_movies = movies_df[movies_df['movie_id'].astype(str).isin(stripped_indices)].reset_index()
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
        # if all(v is None or v == 0 for v in all_ratings):
        #     return render_rating_movies(), {"display": "block"}, {"display": "none"}, True

        user_ratings = {
            movies.iloc[idx]['title']: int(rating)
            for idx, rating in enumerate(all_ratings) if rating and rating > 0
        }
        # print(user_ratings)
        newuser_vector = map_user_ratings_to_full_vector(user_ratings)
        # print(f"user chosen {newuser_vector}")
        recommendations = myIBCF(smatrix2, newuser_vector)
        # print(f"receomnadtion returned  {recommendations}")
        recommended_movies = map_indices_to_movies(recommendations, movies)
        # print(f"inside handler {len(recommended_movies)}")

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

