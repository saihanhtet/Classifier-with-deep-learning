import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self, type) -> None:
        self.type = type
        self.df = self.load_dataframe(self.type)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.encoder = LabelEncoder()
        self.countVectorizer = CountVectorizer()

        self.x_train, self.x_test, self.y_train, self.y_test = self.build_model(frame=[
                                                                                'name', 'type'])

    def load_dataframe(self, type):
        if type == 'sex':
            file = 'data/datasets/classify/sex.csv'
        else:
            file = 'data/datasets/classify/classify.csv'
        if os.path.isfile(file):
            dataframe = pd.read_csv(file, encoding='utf-8')
            return dataframe
        else:
            return self.create_dataframe(type)

    def create_dataframe(self, type):
        if type == 'sex':
            males_mm = pd.read_csv(
                'data/datasets/names/my/male.csv', encoding='utf-8')
            females_mm = pd.read_csv(
                'data/datasets/names/my/female.csv', encoding='utf-8')
            males_mm = [(sex, 'M') for sex in males_mm['NAME']]
            females_mm = [(sex, 'F') for sex in females_mm['NAME']]
            labels = sorted(males_mm + females_mm)
            dataframe = pd.DataFrame(labels, columns=['name', 'type'])
            dataframe.to_csv('data/datasets/classify/sex.csv')
            dataframe = dataframe.drop_duplicates(subset=['name'])
            return dataframe

        elif type == 'classify':
            names = pd.read_csv(
                'data/datasets/names/my/names.csv', encoding='utf-8')
            food_names = pd.read_csv(
                'data/datasets/foods/names.csv', encoding='utf-8')
            animal_names = pd.read_csv(
                'data/datasets/animals/names.csv', encoding='utf-8')
            names = [(name, 'person') for name in names['NAME']]
            animal_names = [(name, 'animal')
                            for name in animal_names['name']]
            food_names = [(name, 'food')
                          for name in food_names['NAME']]
            labels = sorted(
                names + animal_names + food_names)
            dataframe = pd.DataFrame(labels, columns=['name', 'type'])
            dataframe.to_csv('data/datasets/classify/classify.csv')
            dataframe = dataframe.drop_duplicates(subset=['name'])
            return dataframe

        else:
            return False

    def build_model(self, frame: list = None):
        labels = self.df[frame[0]]
        targets = self.df[frame[1]]
        targets_encoded = self.encoder.fit_transform(targets)
        self.countVectorizer.fit(labels)
        labels_transfrom = self.countVectorizer.transform(labels).toarray()
        return train_test_split(
            labels_transfrom, targets_encoded, train_size=0.8, random_state=42)

    def train(self):
        model = LogisticRegression()
        model.fit(self.x_train, self.y_train)
        prediction = model.predict(self.x_test)
        prediction_type = self.encoder.inverse_transform(prediction)
        test_names = self.countVectorizer.inverse_transform(self.x_test)
        types = self.encoder.inverse_transform(self.y_test)
        results_df = pd.DataFrame(
            {'predicted': prediction_type, 'Actually': self.df.loc[self.y_test, 'type'].values})
        results_df['test_names'] = [' '.join(seq) for seq in test_names]
        score = model.score(self.x_test, self.y_test)
        print('Accuracy:', score)
        self.save_model(model, f'models/classify/{self.type}.joblib')
        return model

    def load(self):
        loaded_model = self.load_model(f'models/classify/{self.type}.joblib')
        score = loaded_model.score(self.x_test, self.y_test)
        print("Accuracy from loaded model:", score)
        return loaded_model

    def predict(self, word):
        file = os.path.isfile(f'models/classify/{self.type}.joblib')
        if file:
            model = self.load()
        else:
            model = self.train()
        inp = self.countVectorizer.transform([word]).toarray()
        prediction = model.predict(inp)
        prediction_type = self.encoder.inverse_transform(prediction)
        print(prediction_type)
        return prediction_type

    def save_model(self, model, filename):
        joblib.dump(model, filename=filename)
        return True

    def load_model(self, filename):
        return joblib.load(filename=filename)


if __name__ == "__main__":
    obj = Classifier("sex")
    obj.predict('aung aung')
