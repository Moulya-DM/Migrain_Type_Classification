from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# ---------------- HOME ----------------
def home(request):
    return render(request, 'home.html')


# ---------------- REGISTER ----------------
def register_user(request):

    if request.method == 'POST':

        username = request.POST['username']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():

            return render(request, 'register.html', {
                'error': 'Username already exists'
            })

        User.objects.create_user(
            username=username,
            password=password
        )

        return redirect('/login/')

    return render(request, 'register.html')


# ---------------- LOGIN ----------------
def login_user(request):

    if request.method == 'POST':

        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(request, user)

            return redirect('/predict/')

        else:

            return render(request, 'login.html', {
                'error': 'Invalid credentials'
            })

    return render(request, 'login.html')


# ---------------- PREDICT ----------------
@login_required(login_url='/login/')
def predict(request):

    if request.method == "POST":

        try:

            # =================================================
            # USER INPUT
            # =================================================

            Age = float(request.POST.get('Age'))
            Duration = float(request.POST.get('Duration'))
            Frequency = float(request.POST.get('Frequency'))

            Location = float(request.POST.get('Location'))
            Character = float(request.POST.get('Character'))
            Intensity = float(request.POST.get('Intensity'))

            Nausea = int(request.POST.get('Nausea'))
            Vomit = int(request.POST.get('Vomit'))
            Phonophobia = int(request.POST.get('Phonophobia'))
            Photophobia = int(request.POST.get('Photophobia'))
            Visual = int(request.POST.get('Visual'))
            Sensory = int(request.POST.get('Sensory'))
            Dysphasia = int(request.POST.get('Dysphasia'))
            Dysarthria = int(request.POST.get('Dysarthria'))
            Vertigo = int(request.POST.get('Vertigo'))
            Tinnitus = int(request.POST.get('Tinnitus'))
            Hypoacusis = int(request.POST.get('Hypoacusis'))
            Diplopia = int(request.POST.get('Diplopia'))
            Defect = int(request.POST.get('Defect'))
            Ataxia = int(request.POST.get('Ataxia'))
            Conscience = int(request.POST.get('Conscience'))
            Paresthesia = int(request.POST.get('Paresthesia'))

            DPF = float(request.POST.get('DPF'))

            # =================================================
            # LOAD DATASET
            # =================================================

            path = "migraine_symptom_classification.csv"

            data = pd.read_csv(path)

            # =================================================
            # INPUTS + OUTPUT
            # =================================================

            X = data.drop('Type', axis=1)
            y = data['Type']

            # =================================================
            # ENCODE TARGET LABEL
            # =================================================

            encoder = LabelEncoder()

            y_encoded = encoder.fit_transform(y)

            # =================================================
            # TRAIN TEST SPLIT
            # =================================================

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=0.2,
                random_state=42
            )

            # =================================================
            # LOGISTIC REGRESSION
            # =================================================

            lr_model = LogisticRegression(max_iter=1000)

            lr_model.fit(X_train, y_train)

            lr_pred = lr_model.predict(X_test)

            lr_accuracy = accuracy_score(y_test, lr_pred) * 100

            # =================================================
            # DECISION TREE
            # =================================================

            dt_model = DecisionTreeClassifier()

            dt_model.fit(X_train, y_train)

            dt_pred = dt_model.predict(X_test)

            dt_accuracy = accuracy_score(y_test, dt_pred) * 100

            # =================================================
            # RANDOM FOREST
            # =================================================

            rf_model = RandomForestClassifier(n_estimators=200)

            rf_model.fit(X_train, y_train)

            rf_pred = rf_model.predict(X_test)

            rf_accuracy = accuracy_score(y_test, rf_pred) * 100

            # =================================================
            # KNN
            # =================================================

            knn_model = KNeighborsClassifier(n_neighbors=5)

            knn_model.fit(X_train, y_train)

            knn_pred = knn_model.predict(X_test)

            knn_accuracy = accuracy_score(y_test, knn_pred) * 100

            # =================================================
            # BEST MODEL
            # =================================================

            accuracies = {
                "Logistic Regression": lr_accuracy,
                "Decision Tree": dt_accuracy,
                "Random Forest": rf_accuracy,
                "KNN": knn_accuracy
            }

            best_model_name = max(
                accuracies,
                key=accuracies.get
            )

            # =================================================
            # TEST INPUT
            # =================================================

            test_input = [[
                Age,
                Duration,
                Frequency,
                Location,
                Character,
                Intensity,
                Nausea,
                Vomit,
                Phonophobia,
                Photophobia,
                Visual,
                Sensory,
                Dysphasia,
                Dysarthria,
                Vertigo,
                Tinnitus,
                Hypoacusis,
                Diplopia,
                Defect,
                Ataxia,
                Conscience,
                Paresthesia,
                DPF
            ]]

            # =================================================
            # FINAL PREDICTION
            # =================================================

            if best_model_name == "Logistic Regression":

                final_prediction = lr_model.predict(test_input)

            elif best_model_name == "Decision Tree":

                final_prediction = dt_model.predict(test_input)

            elif best_model_name == "Random Forest":

                final_prediction = rf_model.predict(test_input)

            else:

                final_prediction = knn_model.predict(test_input)

            # =================================================
            # DECODE PREDICTION
            # =================================================

            prediction_label = encoder.inverse_transform(
                final_prediction
            )[0]

            # =================================================
            # SEND RESULT
            # =================================================

            return render(request, 'result.html', {

                'prediction': prediction_label,

                'lr': round(lr_accuracy, 2),
                'dt': round(dt_accuracy, 2),
                'rf': round(rf_accuracy, 2),
                'knn': round(knn_accuracy, 2),

                'best': best_model_name
            })

        except Exception as e:

            print("ERROR:", e)

            return render(request, 'result.html', {
                'prediction': str(e)
            })

    return render(request, 'predict.html')