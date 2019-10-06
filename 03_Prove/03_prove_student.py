from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from sklearn.metrics import r2_score

names = ["school","sex","age","address","fam_size","p_status"
        ,"m_edu","f_edu","m_job","f_job","reason","guardian"
        ,"trvl_time","stdy_time","failures","school_sup","fam_sup"
        ,"paid","activities","nursery","higher","internet","romantic"
        ,"fam_rel","free_time","go_out","d_alc","w_alc","health",
        "absences","G1","G2","G3"]


data = pd.read_csv("03_Prove/student-mat.csv", header=0, skipinitialspace=True, delimiter=';',
                   names=names, na_values=["?"])

# get rid of the first line of headers
data = data.iloc[1:]

# normalize data
data["school"] = data.school.map({"GP": 0, "MS": 1})
data["sex"] = data.sex.map({"M": 0, "F": 1})
data["address"] = data.address.map({"R": 0, "U": 1})
data["fam_size"] = data.fam_size.map({"LE3": 0, "GT3": 1})
data["p_status"] = data.p_status.map({"A": 0, "T": 1})
data["m_job"] = data.m_job.map({"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4})
data["f_job"] = data.f_job.map({"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4})
data["reason"] = data.reason.map({"home": 0, "reputation": 1, "course": 2, "other": 3})
data["guardian"] = data.guardian.map({"mother": 0, "father": 1, "other": 2})
data["school_sup"] = data.school_sup.map({"no": 0, "yes": 1})
data["fam_sup"] = data.fam_sup.map({"no": 0, "yes": 1})
data["paid"] = data.paid.map({"no": 0, "yes": 1})
data["activities"] = data.activities.map({"no": 0, "yes": 1})
data["nursery"] = data.nursery.map({"no": 0, "yes": 1})
data["higher"] = data.higher.map({"no": 0, "yes": 1})
data["internet"] = data.internet.map({"no": 0, "yes": 1})
data["romantic"] = data.romantic.map({"no": 0, "yes": 1})
data["age"] = data.age.astype(float)
data["m_edu"] = data.m_edu.astype(float)
data["f_edu"] = data.f_edu.astype(float)
data["trvl_time"] = data.trvl_time.astype(float)
data["stdy_time"] = data.stdy_time.astype(float)
data["failures"] = data.failures.astype(float)
data["fam_rel"] = data.fam_rel.astype(float)
data["free_time"] = data.free_time.astype(float)
data["go_out"] = data.go_out.astype(float)
data["d_alc"] = data.d_alc.astype(float)
data["w_alc"] = data.w_alc.astype(float)
data["health"] = data.health.astype(float)
data["absences"] = data.absences.astype(float)
data["G1"] = data.G1.astype(float)
data["G2"] = data.G2.astype(float)
data["G3"] = data.G3.astype(float)

# run predictions
data_d = data.drop(columns=["G3"]).values
targ_flat = data["G3"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(data_d, targ_flat, test_size=0.75)
regr = KNeighborsRegressor(n_neighbors=10)
regr.fit(X_train, y_train)

predictions = regr.predict(X_test)
accuracy = r2_score(y_test, predictions)
print("{:.2%}".format(accuracy))