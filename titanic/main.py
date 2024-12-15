import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_data():
    train = pd.read_csv("data/titanic/train.csv")
    test_data = pd.read_csv("data/titanic/test.csv")

    train_target = train[["Survived"]]
    train_data = train.drop(["Survived"], axis=1)

    return train_data, train_target, test_data


def preprocess_data(data):
    # Drop features that are assumed to be irrelevant
    data = data.drop(["Name"], axis=1)
    data = data.drop(["Ticket"], axis=1)
    data = data.drop(["Cabin"], axis=1)

    # Convert sex to binary
    data.loc[data.Sex == 'male', 'Sex'] = 1
    data.loc[data.Sex == 'female', 'Sex'] = 0

    # Convert embarked to numerical
    data.loc[data.Embarked == 'S', 'Embarked'] = 1
    data.loc[data.Embarked == 'C', 'Embarked'] = 2
    data.loc[data.Embarked == 'Q', 'Embarked'] = 3

    return data


def train_model(train_data, train_target):
    model = RandomForestClassifier()
    # Use .values.ravel() to transform target into a 1D array
    model.fit(train_data, train_target.values.ravel())

    return model


def predict_test(model, test_data):
    outcome = model.predict(test_data)

    return outcome


def format_outcome(preprocessed_test_data, test_outcome):
    predictions = pd.DataFrame(preprocessed_test_data.PassengerId)
    predictions['Survived'] = test_outcome

    return predictions


if __name__ == "__main__":
    train_data, train_target, test_data = load_data()
    preprocessed_train_data = preprocess_data(train_data)
    preprocessed_test_data = preprocess_data(test_data)
    model = train_model(preprocessed_train_data, train_target)
    test_outcome = predict_test(model, preprocessed_test_data)
    predictions = format_outcome(preprocessed_test_data, test_outcome)

    predictions.to_csv("data/titanic/predictions.csv", index=False)
