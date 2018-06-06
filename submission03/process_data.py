import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge


class ProcessData:
    train_data_ratio = 0.7

    def __init__(self):
        self.df_train = pd.read_csv('../train.csv', index_col='Id')
        self.df_test = pd.read_csv('../test.csv', index_col='Id')

        # ids of full training dataset
        self.id_train = self.df_train.index

        # ids of full test dataset
        self.id_test = self.df_test.index

    def get_cols_with_na(self, df):
        cols_with_na = df.isnull().sum()
        cols_with_na = cols_with_na[cols_with_na > 0]
        return cols_with_na

    # function to normalise a column of values to lie between 0 and 1
    def scale_minmax(self, col):
        return (col - col.min()) / (col.max() - col.min())

    def generate_clean_data(self):
        # combine train and test datas in to one dataframe
        df_all = pd.concat([self.df_train, self.df_test])
        print('train.shape=', self.df_train.shape, ', test.shape=', self.df_test.shape)


        cols_with_na = self.get_cols_with_na(df_all.drop('SalePrice', axis=1))
        print(cols_with_na.sort_values(ascending=False).to_string())

        # 1.Meaningful NaN Values #########
        # columns where NaN values have meaning e.g. no pool etc.
        cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                       'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType',
                       'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']

        # replace 'NaN' with 'None' in these columns
        for col in cols_fillna:
            df_all[col].fillna('None', inplace=True)

        # GarageYrBlt nans: no garage. Fill with property YearBuilt.
        # (more appropriate than 0, which would be ~2000 away from all other values)
        df_all.loc[df_all.GarageYrBlt.isnull(), 'GarageYrBlt'] = df_all.loc[df_all.GarageYrBlt.isnull(), 'YearBuilt']

        # No masonry veneer - fill area with 0
        df_all.MasVnrArea.fillna(0, inplace=True)

        # No basement - fill areas/counts with 0
        df_all.BsmtFullBath.fillna(0, inplace=True)
        df_all.BsmtHalfBath.fillna(0, inplace=True)
        df_all.BsmtFinSF1.fillna(0, inplace=True)
        df_all.BsmtFinSF2.fillna(0, inplace=True)
        df_all.BsmtUnfSF.fillna(0, inplace=True)
        df_all.TotalBsmtSF.fillna(0, inplace=True)

        # No garage - fill areas/counts with 0
        df_all.GarageArea.fillna(0, inplace=True)
        df_all.GarageCars.fillna(0, inplace=True)

        # 2.LotFrontage NaN Values #########
        # LotFrontage
        # fill nan values using a linear regressor

        # convert categoricals to dummies, exclude SalePrice from model
        df_frontage = pd.get_dummies(df_all.drop('SalePrice', axis=1))

        # normalise columns to 0-1
        for col in df_frontage.drop('LotFrontage', axis=1).columns:
            df_frontage[col] = self.scale_minmax(df_frontage[col])

        lf_train = df_frontage.dropna()
        lf_train_y = lf_train.LotFrontage
        lf_train_X = lf_train.drop('LotFrontage', axis=1)

        # fit model
        lr = Ridge()
        lr.fit(lf_train_X, lf_train_y)

        # check model results
        lr_coefs = pd.Series(lr.coef_, index=lf_train_X.columns)

        print('----------------')
        print('Intercept:', lr.intercept_)
        print('----------------head(10)')
        print(lr_coefs.sort_values(ascending=False).head(10))
        print('----------------tail(10)')
        print(lr_coefs.sort_values(ascending=False).tail(10))
        print('----------------')
        print('R2:', lr.score(lf_train_X, lf_train_y))
        print('----------------')

        # fill na values using model predictions
        nan_frontage = df_all.LotFrontage.isnull()
        X = df_frontage[nan_frontage].drop('LotFrontage', axis=1)
        y = lr.predict(X)

        # fill nan values
        df_all.loc[nan_frontage, 'LotFrontage'] = y

        # 3.Remaining NaNs #########
        print(cols_with_na.sort_values(ascending=False).to_string())

        rows_with_na = df_all.drop('SalePrice', axis=1).isnull().sum(axis=1)
        rows_with_na = rows_with_na[rows_with_na > 0]
        print(rows_with_na.sort_values(ascending=False).to_string())

        # fill remaining nans with mode in that column
        for col in cols_with_na.index:
            df_all[col].fillna(df_all[col].mode()[0], inplace=True)
        # check nans
        df_all.drop('SalePrice', axis=1).isnull().sum().max()

        # Now no more NaN values
        df_all.info()

        # 4.Basement Finish Types #########
        # create separate columns for area of each possible
        # basement finish type
        bsmt_fin_cols = ['BsmtGLQ', 'BsmtALQ', 'BsmtBLQ',
                         'BsmtRec', 'BsmtLwQ']

        for col in bsmt_fin_cols:
            # initialise as columns of zeros
            df_all[col + 'SF'] = 0

        # fill remaining finish type columns
        for row in df_all.index:
            fin1 = df_all.loc[row, 'BsmtFinType1']
            if (fin1 != 'None') and (fin1 != 'Unf'):
                # add area (SF) to appropriate column
                df_all.loc[row, 'Bsmt' + fin1 + 'SF'] += df_all.loc[row, 'BsmtFinSF1']

            fin2 = df_all.loc[row, 'BsmtFinType2']
            if (fin2 != 'None') and (fin2 != 'Unf'):
                df_all.loc[row, 'Bsmt' + fin2 + 'SF'] += df_all.loc[row, 'BsmtFinSF2']

        # remove initial BsmtFin columns
        df_all.drop(['BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2'], axis=1, inplace=True)

        # already have BsmtUnf column in dataset
        bsmt_fin_cols.append('BsmtUnf')

        # also create features representing the fraction of the basement that is each finish type
        for col in bsmt_fin_cols:
            df_all[col + 'Frac'] = df_all[col + 'SF'] / df_all['TotalBsmtSF']
            # replace any nans with zero (for properties without a basement)
            df_all[col + 'Frac'].fillna(0, inplace=True)

        # check nans
        df_all.drop('SalePrice', axis=1).isnull().sum().max()

        # 5.1st and 2nd Floor Area #########
        df_all['LowQualFinFrac'] = df_all['LowQualFinSF'] / df_all['GrLivArea']
        df_all['1stFlrFrac'] = df_all['1stFlrSF'] / df_all['GrLivArea']
        df_all['2ndFlrFrac'] = df_all['2ndFlrSF'] / df_all['GrLivArea']

        df_all['TotalAreaSF'] = df_all['GrLivArea'] + df_all['TotalBsmtSF'] + df_all['GarageArea'] + df_all['EnclosedPorch'] + \
                                df_all['ScreenPorch']
        df_all['LivingAreaSF'] = df_all['1stFlrSF'] + df_all['2ndFlrSF'] + df_all['BsmtGLQSF'] + df_all['BsmtALQSF'] + df_all[
            'BsmtBLQSF']
        df_all['StorageAreaSF'] = df_all['LowQualFinSF'] + df_all['BsmtRecSF'] + df_all['BsmtLwQSF'] + df_all['BsmtUnfSF'] + \
                                  df_all['GarageArea']

        # 6.Categorical Features with Meaningful Ordering #########
        # convert some categorical values to numeric scales

        # Excellent, Good, Typical, Fair, Poor, None: Convert to 0-5 scale
        cols_ExGd = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                     'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                     'GarageCond', 'PoolQC']

        dict_ExGd = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}

        for col in cols_ExGd:
            df_all[col].replace(dict_ExGd, inplace=True)

        print(df_all[cols_ExGd].head(5))

        # Remaining columns
        df_all['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}, inplace=True)

        df_all['CentralAir'].replace({'Y': 1, 'N': 0}, inplace=True)

        df_all['Functional'].replace({'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0},
                                     inplace=True)

        df_all['GarageFinish'].replace({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}, inplace=True)

        df_all['LotShape'].replace({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}, inplace=True)

        df_all['Utilities'].replace({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0}, inplace=True)

        df_all['LandSlope'].replace({'Gtl': 2, 'Mod': 1, 'Sev': 0}, inplace=True)

        # 7.Dealing with Zeros #########
        # fraction of zeros in each column
        frac_zeros = ((df_all==0).sum()/len(df_all))

        # no. unique values in each column
        n_unique = df_all.nunique()

        # difference between frac. zeros and expected
        # frac. zeros if values evenly distributed between
        # classes
        xs_zeros = frac_zeros - 1/n_unique

        # create dataframe and display which columns may be problematic
        zero_cols = pd.DataFrame({'frac_zeros':frac_zeros,'n_unique':n_unique,'xs_zeros':xs_zeros})
        zero_cols = zero_cols[zero_cols.frac_zeros>0]
        zero_cols.sort_values(by='xs_zeros',ascending=False,inplace=True)
        print(zero_cols[(zero_cols.xs_zeros>0)])

        #very few properties with Pool or 3SsnPorch
        #replace columns with binary indicator
        df_all['HasPool'] = (df_all['PoolQC']>0).astype(int)
        df_all['Has3SsnPorch'] = (df_all['3SsnPorch']>0).astype(int)
        df_all.drop(['PoolQC','PoolArea','3SsnPorch'],axis=1,inplace=True)

        # 'half' bathrooms - add half value to 'full' bathrooms
        df_all['BsmtFullBath'] = df_all['BsmtFullBath'] + 0.5*df_all['BsmtHalfBath']
        df_all['FullBath'] = df_all['FullBath'] + 0.5*df_all['HalfBath']
        df_all.drop(['BsmtHalfBath','HalfBath'],axis=1,inplace=True)

        # create additional dummy variable for
        # continuous variables with a lot of zeros
        dummy_cols = ['LowQualFinSF','2ndFlrSF',
                      'MiscVal','ScreenPorch','WoodDeckSF','OpenPorchSF',
                      'EnclosedPorch','MasVnrArea','GarageArea','Fireplaces',
                      'BsmtGLQSF','BsmtALQSF','BsmtBLQSF','BsmtRecSF',
                      'BsmtLwQSF','BsmtUnfSF','TotalBsmtSF']

        for col in dummy_cols:
            df_all['Has'+col] = (df_all[col]>0).astype(int)

        # 8.Log Transform SalePrice #########
        #Log Transform SalePrice to improve normality
        sp = df_all.SalePrice
        df_all.SalePrice = np.log(sp)

        print(df_all.SalePrice.describe())

        # 9.Identify Types of Features #########
        # extract names of numeric columns
        dtypes = df_all.dtypes
        cols_numeric = dtypes[dtypes != object].index.tolist()

        # MSubClass should be treated as categorical
        cols_numeric.remove('MSSubClass')

        # choose any numeric column with less than 13 values to be
        # "discrete". 13 chosen to include months of the year.
        # other columns "continuous"
        col_nunique = dict()

        for col in cols_numeric:
            col_nunique[col] = df_all[col].nunique()

        col_nunique = pd.Series(col_nunique)

        cols_discrete = col_nunique[col_nunique < 13].index.tolist()
        cols_continuous = col_nunique[col_nunique >= 13].index.tolist()

        print(len(cols_numeric), 'numeric columns, of which',
              len(cols_continuous), 'are continuous and',
              len(cols_discrete), 'are discrete.')

        # extract names of categorical columns
        cols_categ = dtypes[~dtypes.index.isin(cols_numeric)].index.tolist()

        for col in cols_categ:
            df_all[col] = df_all[col].astype('category')

        print(len(cols_categ), 'categorical columns.')

        # 10.Correlation Between Numeric Features #########
        # correlation between numeric variables
        df_corr = df_all.loc[self.id_train, cols_numeric].corr(method='spearman').abs()

        # order columns and rows by correlation with SalePrice
        df_corr = df_corr.sort_values('SalePrice',axis=0,ascending=False).sort_values('SalePrice',axis=1,ascending=False)

        print(df_corr.SalePrice.head(20))
        print('-----------------')
        print(df_corr.SalePrice.tail(10))

        # 11.Normalise Numeric Features #########
        # normalise numeric columns
        scale_cols = [col for col in cols_numeric if col!='SalePrice']
        df_all[scale_cols] = df_all[scale_cols].apply(self.scale_minmax,axis=0)
        df_all[scale_cols].describe()

        # 12.Box-Cox Transform Suitable Variables #########
        # variables not suitable for box-cox transformation based on above (usually due to excessive zeros)
        cols_notransform = ['2ndFlrSF', '1stFlrFrac', '2ndFlrFrac', 'StorageAreaSF',
                            'EnclosedPorch', 'LowQualFinSF', 'MasVnrArea',
                            'MiscVal', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF', 'SalePrice',
                            'BsmtGLQSF', 'BsmtALQSF', 'BsmtBLQSF', 'BsmtRecSF', 'BsmtLwQSF', 'BsmtUnfSF',
                            'BsmtGLQFrac', 'BsmtALQFrac', 'BsmtBLQFrac', 'BsmtRecFrac', 'BsmtLwQFrac', 'BsmtUnfFrac']

        cols_transform = [col for col in cols_continuous if col not in cols_notransform]

        # transform remaining variables
        print('Transforming', len(cols_transform), 'columns:', cols_transform)

        for col in cols_transform:
            # transform column
            df_all.loc[:, col], _ = stats.boxcox(df_all.loc[:, col] + 1)

            # renormalise column
            df_all.loc[:, col] = self.scale_minmax(df_all.loc[:, col])

        # 13.Prepare Data for Model Fitting #########
        # select features, encode categoricals, create dataframe for model fitting

        # select which features to use (all for now)
        model_cols = df_all.columns

        # encode categoricals
        self.df_model = pd.get_dummies(df_all[model_cols])

        # Rather than including Condition1 and Condition2, or Exterior1st and Exterior2nd,
        # combine the dummy variables (allowing 2 true values per property)
        if ('Condition1' in model_cols) and ('Condition2' in model_cols):

            cond_suffix = ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNn']

            for suffix in cond_suffix:
                col_cond1 = 'Condition1_' + suffix
                col_cond2 = 'Condition2_' + suffix

                self.df_model[col_cond1] = self.df_model[col_cond1] | self.df_model[col_cond2]
                self.df_model.drop(col_cond2, axis=1, inplace=True)

        if ('Exterior1st' in model_cols) and ('Exterior2nd' in model_cols):

            # some different strings in Exterior1st and Exterior2nd for same type - rename columns to correct
            self.df_model.rename(columns={'Exterior2nd_Wd Shng': 'Exterior2nd_WdShing',
                                     'Exterior2nd_Brk Cmn': 'Exterior2nd_BrkComm',
                                     'Exterior2nd_CmentBd': 'Exterior2nd_CemntBd'}, inplace=True)

            ext_suffix = ['AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd',
                          'HdBoard', 'ImStucc', 'MetalSd', 'Plywood', 'Stone',
                          'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'AsbShng']

            for suffix in ext_suffix:
                col_cond1 = 'Exterior1st_' + suffix
                col_cond2 = 'Exterior2nd_' + suffix

                self.df_model[col_cond1] = self.df_model[col_cond1] | self.df_model[col_cond2]
                self.df_model.drop(col_cond2, axis=1, inplace=True)

        print(self.df_model.head())

    # function to get training samples
    def get_training_data(self):
        # extract training samples
        df_train = self.df_model.loc[self.id_train]
        df_train = df_train.loc[:df_train.shape[0] * ProcessData.train_data_ratio]

        # split SalePrice and features
        y = df_train.SalePrice
        X = df_train.drop('SalePrice', axis=1)

        return X, y

    # function to get validation samples
    def get_validation_data(self):
        # extract validation samples
        df_train = self.df_model.loc[self.id_train]
        df_train = df_train.loc[df_train.shape[0] * ProcessData.train_data_ratio:]

        # split SalePrice and features
        y = df_train.SalePrice
        X = df_train.drop('SalePrice', axis=1)

        return X, y

    # extract test data (without SalePrice)
    def get_test_data(self):
        return self.df_model.loc[self.id_test].drop('SalePrice', axis=1)




