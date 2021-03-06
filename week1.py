import pandas
from collections import Counter

def strip_name(name):
	if name.find('Miss.') > -1:
		return name[name.find('Miss')+6:].split(' ', 1)[0]
	elif name.find('Mrs') > -1:
			if name.find('(') > -1:
				return name[name.find('(')+1:len(name)-1].split(' ', 1)[0]
		

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
print (data['Sex'].value_counts())
print('-' * 40)
perc_survived = data['Survived'].sum() * 100 / data['Survived'].count()
perc_survived = round(perc_survived,2)
print (perc_survived, 100 - perc_survived)
print('-' * 40)
Pclass_counts = data['Pclass'].value_counts()
perc_first = round(Pclass_counts[1]  * 100 / (Pclass_counts[1] +  Pclass_counts[2] + Pclass_counts[3]),2)
print(perc_first)
print('-' * 40)
age_mean = round(data['Age'].mean(),2)
age_median = data['Age'].median()
print(age_mean, age_median)
print('-' * 40)
df = data[['SibSp', 'Parch']].corr(method='pearson')
print(df)
print('-' * 40)
females = data.ix[data.Sex == 'female']
females_names = females['Name']

first_names = []
for i in females_names:
	pass
	#print(strip_name(i))
	first_names.append(strip_name(i))
print(Counter(first_names))

