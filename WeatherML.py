import pyrasgo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#your personal API key from RasgoML.com - do not share
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjE2NiwiaWQiOjE2Niwib3JnSWQiOjEsInVzZXJuYW1lIjoiamVkYmVsbDcwNTBAZ21haWwuY29tIiwiaWF0IjoxNjAxOTI3OTQ2LCJleHAiOjE2MDU1Mjc5NDZ9.WOPfYjciDJHIb8zkvUNsa1N3sdj1LTj9RRx52Hw8fnQ'

#authenticate your API client
rasgo = pyrasgo.connect(api_key)

#print all of the models you have created through RasgoML.com
models = rasgo.get_models() 

#print all of your model IDs and descriptions
print('Current models: {}'.format(models))

#paste your model ID into the quotes to pull model training data for that model
model_ID = '342'

#load feature data for the model you want to train into a pandas dataframe
#limit of 10 is included for purposes of testing and can be removed when you're ready
df = rasgo.get_feature_data(model_ID, limit = 100000)
# print(df)
# print(df.columns)
df = df.dropna()
X = df.drop(['DATE', 'FIPS', 'DS_WEATHER_ICON'], axis='columns')
y = df.DS_WEATHER_ICON

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred) 
print(cm)

plt.figure()
ax1 = sns.heatmap(cm, yticklabels=['clear-day', 'cloudy', 'partly-cloudy', 'rain', 'snow', 'wind'], xticklabels=['clear-day', 'cloudy', 'partly-cloudy', 'rain', 'snow', 'wind'], cmap="YlGnBu", annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
 
plt.figure()
ax2 = sns.barplot(model.feature_importances_, X.columns)
plt.xlabel('Importance')
plt.ylabel('Feature')


plt.show()