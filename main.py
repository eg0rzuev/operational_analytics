import os
import json
import sys
import numpy
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler

numpy.set_printoptions(threshold=sys.maxsize)

model_results = {}
song_artist = {'jingle_bell_rock': ['christmas', 'bobby_helms'],
               'last_christmas': ['christmas'],
               'jingle_bells': ['christmas'],
               'usa_anthem': ['thanksgiving', 'independence', 'washingtons_bday', 'memorial_day'],
               'america_the_beautiful': ['thanksgiving', 'independence', 'washingtons_bday', 'memorial_day', '9_11'],
               'god_bless_america': ['independence', 'washingtons_bday', 'memorial_day', '9_11'],
               'santa_tell_me': ['ariana_grande', 'christmas']}


def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:i + look_back, 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x).astype('float32'), np.array(data_y).astype('float32')


def read_file(filename):
    # read from the file
    df = pd.read_csv(filename, usecols=[0, 1], skiprows=1)

    # basic preprocessing. Replace <1 to 1 for logarithm function & to make the data transformable to float32 type
    df['popularity'] = df['popularity'].replace('<1', 1).astype(np.float32)
    song_name, _ = os.path.splitext(os.path.basename(complete_filename))
    print(song_name)

    # using the algorithms
    linear_regression(df, song_name)
    linear_regression_simple(df, song_name)
    sarimax(df, song_name)
    svr(df, song_name)
    svr(df, song_name, use_exogenous=True)
    sequential(df, song_name)
    sarima(df, song_name)
    return


def sequential(df, song_name):
    df['popularity'] = df['popularity'].replace(0, 1)
    dataset = df.values

    train_size = int(len(dataset) * 0.6)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size

    # get rid of zeroes for log transform
    train, valid, test = (np.log(dataset[0:train_size, 1:].astype('float64')),
                          np.log(dataset[train_size:train_size + valid_size, 1:].astype('float64')),
                          np.log(dataset[train_size + valid_size:len(dataset), 1:].astype('float64')))
    look_back = 52

    train_x, train_y = create_dataset(train, look_back)
    val_x, val_y = create_dataset(valid, look_back)

    # grid search using for-loops
    hidden_neurons_values = [4, 8, 16]
    loss_functions = ['mean_squared_error', 'mean_absolute_error']
    batch_sizes = [2, 4, 8]

    best_model = None
    best_score = float('inf')

    for hidden_neurons in hidden_neurons_values:
        for loss_function in loss_functions:
            for batch_size in batch_sizes:
                model = Sequential()
                model.add(Dense(hidden_neurons, input_dim=52, activation='relu'))
                model.add(Dense(1))
                model.compile(loss=loss_function, optimizer='adam')
                model.fit(train_x, train_y, epochs=200, batch_size=batch_size, verbose=0)

                loss = model.evaluate(val_x, val_y)
                if loss < best_score:
                    best_score = loss
                    best_model = model

    curr = numpy.array([train_x[-1]])
    test_forecast = []
    count = 0
    while count < test_size:
        new_el = best_model.predict(curr)[0][0]
        test_forecast.append(new_el)
        curr[0] = numpy.append(numpy.delete(curr[0], [0]), [new_el])
        count += 1
    test_forecast = np.array([np.array([x]) for x in test_forecast])

    plot_data(dataset, np.arange(train_size + valid_size).reshape(-1, 1),
              np.arange(train_size + valid_size, train_size + test_size + valid_size).reshape(-1, 1),
              dataset[0:train_size + valid_size, 1:], np.exp(test_forecast), song_name, 'sequential')
    evaluate_model(np.exp(test), np.exp(test_forecast), song_name, 'sequential')


def linear_regression_simple(df, song_name):
    dataset = df.values
    # train, test datasets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, 1], dataset[train_size:len(dataset), 1]
    regression = LinearRegression().fit(np.arange(train_size).reshape(-1, 1), train)
    train_pred = regression.predict(np.arange(train_size).reshape(-1, 1))
    test_forecast = regression.predict(np.arange(train_size, train_size + test_size).reshape(-1, 1))
    plot_data(dataset, np.arange(train_size).reshape(-1, 1), np.arange(train_size, train_size + test_size),
              train_pred, test_forecast, song_name, 'lin_reg_simple')
    evaluate_model(test, test_forecast, song_name, 'lin_reg_simple')


def linear_regression(df, song_name):
    df['popularity'] = df['popularity'].replace(0, 1)
    dataset = df.values
    # binary encoding preprocessing
    week_columns = np.zeros((dataset.shape[0], 52))

    # numerate weeks
    week = 0
    for i in range(dataset.shape[0]):
        week_columns[i, week] = 1
        week += 1
        week %= 52

    # append week array to the dataset
    dataset = np.column_stack((dataset, week_columns))

    # train, test datasets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    # temp = dataset[:, 1]
    # temp[temp <= 1] = 1
    # dataset[:, 1] = temp
    # dataset[:, 1] = dataset[:, 1].astype('float64')

    train, test = (np.log(dataset[0:train_size, 1].astype('float64')),
                   np.log(dataset[train_size:len(dataset), 1].astype('float64')))
    train_x = dataset[0:train_size, 2:]

    regression = LinearRegression().fit(train_x, train)
    train_pred = regression.predict(train_x)
    test_x = dataset[train_size:len(dataset), 2:]
    test_forecast = regression.predict(test_x)

    # plot data
    plot_data(dataset, np.arange(train_size).reshape(-1, 1), np.arange(train_size, train_size + test_size),
              np.exp(train_pred), np.exp(test_forecast), song_name, 'lin_reg')

    # evaluate the model
    evaluate_model(np.exp(test), np.exp(test_forecast), song_name, 'lin_reg')


def sarima(df, song_name):
    dataset = df.values
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, 1], dataset[train_size:len(dataset), 1]
    model = auto_arima(train, start_p=1, start_q=1,
                       test='adf', max_p=5, max_q=5, m=52,
                       start_P=0, seasonal=True,
                       d=None, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)
    test_forecast, confint = model.predict(test.shape[0], return_conf_int=True)
    plot_data(dataset, np.arange(train_size).reshape(-1, 1), np.arange(train_size, train_size + test_size),
              train, test_forecast, song_name, 'sarima', confint=confint)
    evaluate_model(test, test_forecast, song_name, 'sarima')


def sarimax(df, song_name):
    if song_name not in song_artist.keys():
        return
    dataset = df.values
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, 1], dataset[train_size:len(dataset), 1]

    # obtain exogenous data
    exogenous = []
    for artist_data in song_artist[song_name]:
        path = os.path.join('csv_popularity', artist_data + '.csv')
        temp = pd.read_csv(path, usecols=[0, 1], skiprows=1)
        temp['popularity'] = temp['popularity'].replace('<1', 1).astype(np.float32)
        temp['popularity'] = temp['popularity'].diff().bfill()
        popularity_data = temp['popularity'].to_numpy()
        exogenous.append(popularity_data)

    exogenous = np.array(exogenous)
    exogenous = np.transpose(exogenous)
    model = auto_arima(train, start_p=1, start_q=1, X=exogenous[0:train_size, :],
                       test='adf', max_p=5, max_q=5, m=52,
                       start_P=0, seasonal=True,
                       d=None, D=1, trace=True,
                       error_action='trace',
                       suppress_warnings=True,
                       stepwise=True)

    test_forecast, confint = model.predict(test.shape[0], X=exogenous[train_size:, :], return_conf_int=True)

    plot_data(dataset, np.arange(train_size).reshape(-1, 1), np.arange(train_size, train_size + test_size),
              train, test_forecast, song_name, 'sarimax', confint=confint)

    evaluate_model(test, test_forecast, song_name, 'sarimax')


def svr(df, song_name, use_exogenous=False):
    dataset = df.values
    week_columns = np.zeros((dataset.shape[0], 52))

    # iterate the dataset and set the week number
    week = 0
    for i in range(dataset.shape[0]):
        week_columns[i, week] = 1
        week += 1
        week %= 52

    # append the array for week columns to the original dataset
    dataset = np.column_stack((dataset, week_columns))
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    if use_exogenous:
        # Append exogenous data
        exogenous = []
        for artist_data in song_artist[song_name]:
            sc_x = StandardScaler()
            path = os.path.join('csv_popularity', artist_data + '.csv')
            temp = pd.read_csv(path, usecols=[0, 1], skiprows=1)
            temp['popularity'] = temp['popularity'].replace('<1', 1).astype(np.float32)
            temp['popularity'] = temp['popularity'].diff().bfill()
            popularity_data = temp['popularity'].to_numpy().reshape(-1, 1)
            popularity_data = sc_x.fit_transform(popularity_data)
            exogenous.append(popularity_data)

        exogenous = np.array(exogenous)
        exogenous = np.transpose(exogenous)
        dataset = np.column_stack((dataset, exogenous[0]))

    sc_y = StandardScaler()

    y = dataset[:, 1].reshape(-1, 1)
    y = sc_y.fit_transform(y)
    train, test = y[0:train_size].flatten(), y[train_size:len(dataset)]
    train_x = dataset[0:train_size, 2:]
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],  # You can add other kernel options
        'C': [0.1, 1, 10],  # Regularization parameter
        'epsilon': [0.01, 0.1, 1],  # Epsilon parameter
    }

    # Create an SVR model
    svr_model = SVR()

    # Create a GridSearchCV object with cross-validation
    grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    # Fit the GridSearchCV object to the training data
    grid_search.fit(train_x, train)
    # Get the best parameters from grid search
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Use the best model for predictions
    best_svr = grid_search.best_estimator_

    test_x = dataset[train_size:len(dataset), 2:]
    test_forecast = best_svr.predict(test_x)
    test_forecast = sc_y.inverse_transform(test_forecast.reshape(-1, 1))

    test = sc_y.inverse_transform(test.reshape(-1, 1))
    train = sc_y.inverse_transform(train.reshape(-1, 1))

    # evaluate the model's performance
    plot_data(dataset, np.arange(train_size).reshape(-1, 1), np.arange(train_size, train_size + test_size),
              train, test_forecast, song_name, 'svr' + ('_exog' if use_exogenous else ''))

    # evaluate the model

    evaluate_model(test, test_forecast, song_name, 'svr' + ('_exog' if use_exogenous else ''))


def plot_data(dataset, train_pred_x, test_forecast_x, train_pred_y, test_forecast_y, plot_name, dir_name, confint=None):
    # filter y data
    test_forecast_y[test_forecast_y > 100] = 100
    test_forecast_y[test_forecast_y < 0] = 0
    # plot the data
    plt.plot(dataset[:, 1])
    plt.plot(train_pred_x, train_pred_y)
    plt.plot(test_forecast_x, test_forecast_y)
    plt.title(plot_name)
    # save the plot
    plot_path = os.path.join('plots', dir_name)
    if confint is not None:
        lower_series = pd.Series(confint[:, 0])
        upper_series = pd.Series(confint[:, 1])
        plt.fill_between(test_forecast_x,
                         lower_series,
                         upper_series,
                         color='k', alpha=.15)

    os.makedirs(plot_path, exist_ok=True)
    plot_path = os.path.join(plot_path, plot_name + ".png")
    plt.savefig(plot_path)
    # show
    plt.show()


def evaluate_model(test, test_forecast, song_name, model_name):
    # filter forecast data
    test_forecast[test_forecast >= 100] = 100
    test_forecast[test_forecast < 0] = 0
    test_mse = mean_squared_error(test, test_forecast)
    test_rmse = np.sqrt(test_mse)

    if song_name in model_results:
        model_results[song_name][model_name] = {}
        model_results[song_name][model_name]['MSE'] = test_mse
    else:
        model_results[song_name] = {}
        model_results[song_name][model_name] = {}
        model_results[song_name][model_name]['MSE'] = test_mse

    print("\nTest Set Metrics:")
    print(f"MSE: {test_mse:.2f}")
    print(f"RMSE: {test_rmse:.2f}")


def plot_mse_comparison_by_song():
    for song_name, models in model_results.items():
        model_names = list(models.keys())
        mse_values = [models[model]['MSE'] for model in model_names]

        # bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, mse_values, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('MSE')
        plt.title(f'MSE Comparison for {song_name}')
        plt.xticks(rotation=45)

        for bar, mse in zip(bars, mse_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{mse:.2f}', ha='center', va='center')

        plt.tight_layout()

        plot_path = os.path.join('plots', 'results')
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(plot_path, song_name + ".png")
        plt.savefig(plot_path)
        plt.show()


directory_path = 'csv_popularity'
for song in song_artist:
    song = song + '.csv'
    complete_filename = os.path.join(directory_path, song)
    if os.path.isfile(complete_filename):
        read_file(complete_filename)

print(model_results)
with open('results.txt', 'w') as f:
    f.write(json.dumps(
        model_results,
        sort_keys=True,
        separators=(',', ': '),
        indent=4
    ))

plot_mse_comparison_by_song()
