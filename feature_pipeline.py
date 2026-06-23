import pandas as pd
import numpy as np

MARKDOWN_COLS = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']


def load_raw_data(path='data/'):
    train = pd.read_csv(path + 'train.csv', parse_dates=['Date'])
    test = pd.read_csv(path + 'test.csv', parse_dates=['Date'])
    features = pd.read_csv(path + 'features.csv', parse_dates=['Date'])
    stores = pd.read_csv(path + 'stores.csv')
    return train, test, features, stores


def merge_all(df, features, stores):
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    return df


def combine_train_test(train, test, features, stores):
    train_df = merge_all(train, features, stores)
    test_df = merge_all(test, features, stores)

    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['Weekly_Sales'] = np.nan

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    return full_df


def process_markdowns(df):
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)
    df['markdown_total'] = df[MARKDOWN_COLS].sum(axis=1)
    df['n_active_markdowns'] = (df[MARKDOWN_COLS] > 0).sum(axis=1)
    return df


def flag_sparse_series(df, train_df, min_weeks=52):
    pair_counts = train_df.groupby(['Store', 'Dept'])['Date'].count().reset_index(name='n_weeks')
    df = df.merge(pair_counts, on=['Store', 'Dept'], how='left')
    df['is_sparse'] = df['n_weeks'] < min_weeks
    return df


def add_calendar_features(df):
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    major_holidays = pd.to_datetime([
        '2010-02-12', '2011-02-11', '2012-02-10',  # Super Bowl
        '2010-09-10', '2011-09-09', '2012-09-07',  # Labor Day
        '2010-11-26', '2011-11-25', '2012-11-23',  # Thanksgiving
        '2010-12-31', '2011-12-30', '2012-12-28',  # Christmas (week bucket)
    ])

    df['days_to_nearest_holiday'] = df['Date'].apply(
        lambda d: min(abs((d - h).days) for h in major_holidays)
    )

    thanksgiving_dates = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23'])
    week_after = [d + pd.Timedelta(weeks=1) for d in thanksgiving_dates]
    df['is_week_after_thanksgiving'] = df['Date'].isin(week_after).astype(int)

    return df


def add_lag_rolling_features(df):
    df = df.sort_values(['Store', 'Dept', 'Date'])
    group = df.groupby(['Store', 'Dept'])['Weekly_Sales']

    for lag in [1, 2, 3, 4]:
        df[f'lag_{lag}'] = group.shift(lag)

    df['lag_52'] = group.shift(52)
    df['lag_53'] = group.shift(53)

    shifted = group.shift(1)
    regroup_keys = [df['Store'], df['Dept']]
    df['rolling_mean_4'] = shifted.groupby(regroup_keys).transform(lambda x: x.rolling(4).mean())
    df['rolling_mean_8'] = shifted.groupby(regroup_keys).transform(lambda x: x.rolling(8).mean())
    df['rolling_std_4'] = shifted.groupby(regroup_keys).transform(lambda x: x.rolling(4).std())
    df['rolling_std_8'] = shifted.groupby(regroup_keys).transform(lambda x: x.rolling(8).std())

    df['yoy_growth'] = (df['lag_1'] - df['lag_53']) / df['lag_53'].replace(0, np.nan)

    return df


def add_group_aggregates(df, train_df):
    dept_avg = train_df.groupby('Dept')['Weekly_Sales'].mean().rename('dept_avg_sales')
    store_avg = train_df.groupby('Store')['Weekly_Sales'].mean().rename('store_avg_sales')

    df = df.merge(dept_avg, on='Dept', how='left')
    df = df.merge(store_avg, on='Store', how='left')
    return df


def add_store_type_ordinal(df):
    type_order = {'A': 3, 'B': 2, 'C': 1}
    df['type_ordinal'] = df['Type'].map(type_order)
    return df


def run_feature_pipeline(train, test, features, stores, min_weeks=52):
    full_df = combine_train_test(train, test, features, stores)
    full_df = process_markdowns(full_df)
    full_df = flag_sparse_series(full_df, train, min_weeks=min_weeks)
    full_df = add_calendar_features(full_df)
    full_df = add_lag_rolling_features(full_df)
    full_df = add_group_aggregates(full_df, train)
    full_df = add_store_type_ordinal(full_df)
    return full_df


def split_and_save(full_df, train_path='processed_train.csv', test_path='processed_test.csv'):
    processed_train = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
    processed_test = full_df[full_df['is_train'] == 0].drop(columns=['is_train', 'Weekly_Sales'])

    processed_train.to_csv(train_path, index=False)
    processed_test.to_csv(test_path, index=False)

    return processed_train, processed_test
