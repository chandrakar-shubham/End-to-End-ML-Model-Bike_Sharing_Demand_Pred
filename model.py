from train import *
from sklearn.pipeline import Pipeline
import pickle

final_pipe = Pipeline([
        ('processor',processor),
        ('xgboost',xgb)
        ])

final_pipe.fit(X_train,y_train)

final_pipe.predict(X_train)

final_pipe.score(X_test,y_test)

final_pipe.predict(X_test)


# pickle the trained pipeline and save it to a file
with open('final_pipe.pkl', 'wb') as file:
    pickle.dump(final_pipe, file)

print('model py file executed')