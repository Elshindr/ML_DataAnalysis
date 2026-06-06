import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

## Which Form is this that ? Wow!! Incr o/!


# Step One: Create Random DataFrame with severals type of forms
def generate_shapes_dataset(n_samples=100):
    sides = []
    right_angles = []
    labels = []
    all_sides_equal = []

    for _ in range(n_samples):

        shape_type = np.random.choice(
            [
                "Carré",
                "Rectangle",
                "Triangle",
                "Cercle",
                "Losange",
                "Triangle Rectangle",
            ]
        )

        if shape_type == "Carré":
            sides.append(4)
            right_angles.append(4)
            all_sides_equal.append(1)

        elif shape_type == "Rectangle":
            sides.append(4)
            right_angles.append(4)
            all_sides_equal.append(0)

        elif shape_type == "Losange":
            sides.append(4)
            right_angles.append(0)
            all_sides_equal.append(1)

        elif shape_type == "Trapéze":
            sides.append(4)
            right_angles.append(0)
            all_sides_equal.append(0)

        elif shape_type == "Triangle":
            sides.append(3)
            right_angles.append(0)  # Triangle régulier ou isocele
            all_sides_equal.append(np.random.choice([0, 1]))

        elif shape_type == "Triangle Rectangle":
            sides.append(3)
            right_angles.append(1)
            all_sides_equal.append(0)

        elif shape_type == "Cercle":
            sides.append(1)
            right_angles.append(0)
            all_sides_equal.append(0)

        labels.append(shape_type)

    df = pd.DataFrame(
        {
            "Nombre_cotes": sides,
            "Nombre_angles_droits": right_angles,
            "Cotes_egaux": all_sides_equal,
            "Forme": labels,
        }
    )

    return df


df_train = generate_shapes_dataset(1000)
print(df_train.head())


# Step Two: Create model
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_absolute_error


#### TEst Best MAx depth
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes

arr_mae = []
for l in candidate_max_leaf_nodes:
    arr_mae.append(get_mae(l, train_X, val_X, train_y, val_y))

df_mae = pd.DataFrame({"mae": arr_mae, "node": candidate_max_leaf_nodes})

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = df_mae["node"][df_mae["mae"].idxmin()]

print(best_tree_size)

model = DecisionTreeClassifier(random_state=3, max_depth=4, ccp_alpha=0)
# model = LinearSVC(random_state=3, dual=False, C=0.5)


# Step Three: Train model
X_train = df_train.drop("Forme", axis=1)
y_train = df_train["Forme"]

model.fit(X_train, y_train)

# Step Four: Test prediction from model
df_test = generate_shapes_dataset(n_samples=100000)
X_test = df_test.drop("Forme", axis=1)
y_test = df_test["Forme"]


# Encoder les étiquettes formes(catégories) en valeurs numériques
le = LabelEncoder()
y_encoded = le.fit_transform(y_test)  # Convertir les catégories en valeurs numériques
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)


print(f"Valeurs encodées: {y_test}")  # Par exemple: [0, 1, 0, 2, 1]
mae = mean_absolute_error(y_test, y_pred)

df_pred = pd.DataFrame(
    {
        "Nombre_cotes": df_test["Nombre_cotes"],
        "Nombre_angles_droits": df_test["Nombre_angles_droits"],
        "Cotes_egaux": df_test["Cotes_egaux"],
        "Forme": y_test,
        "Pred": y_pred,
    }
)

print(df_pred.head())
print(f"Accuracy: {accuracy} :: MAE {mae}")


# Step Five: Show confusion matrix
def show_confusion_matrice(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité")
    plt.title("Matrice de Confusion")
    plt.show()


show_confusion_matrice(y_test, y_pred)


def show_tree_decision():
    from sklearn import tree

    plt.figure(figsize=(10, 10))
    tree.plot_tree(
        model,
        feature_names=X_test.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True,
    )
    plt.show()


show_tree_decision()


# Step Six: Fonction to predict which form it is
def predict_shape(n_sides, n_right_angles, n_equal_side):
    df = pd.DataFrame(
        [[n_sides, n_right_angles, n_equal_side]],
        columns=["Nombre_cotes", "Nombre_angles_droits", "Cotes_egaux"],
    )
    return model.predict(df)[0]


# Step Seven: chatbot loop
def chatbox():
    user_input_sides.lower() == "2"
    print(
        "Bienvenue dans ma chatbox de classification des formes géométriques ! Trop incr!"
    )
    print("Saisir 'exit' pour fuir")

    while user_input_sides.lower() != "exit":

        if user_input_sides.lower() == "exit":
            print("Au revoir!")

        # Nb of size?
        user_input_sides = input("Entrez le nombre de côtés (1, 3 ou 4) : ")
        try:
            n_sides = int(user_input_sides)
            if n_sides not in [1, 3, 4]:
                print(
                    "Erreur : Veuillez entrer 1 pour un cercle, 3 pour un triangle ou 4 pour un carré/rectangle."
                )
                continue
        except ValueError:
            print("Erreur : Veuillez entrer un nombre valide.")
            continue

        # Nb angles
        user_input_right_angles = input("Entrez le nombre d'angles droits (0 à 4) : ")
        if user_input_right_angles.lower() == "exit":
            print("Au revoir!")
            break

        try:
            n_right_angles = int(user_input_right_angles)
            if n_sides == 1 and n_right_angles != 0:
                print("Erreur : Un cercle peut avoir 0 angle droit.")
                continue
            if n_sides == 3 and n_right_angles not in [0, 1]:
                print("Erreur : Un triangle peut avoir 0 ou 1 angle droit.")
                continue
            elif n_sides == 4 and n_right_angles != 4:
                print("Erreur : Un carré ou un rectangle doit avoir 4 angles droits.")
                continue
        except ValueError:
            print("Erreur : Veuillez entrer un nombre valide.")
            continue

        # Is side egal ?
        user_input_all_sides_equal = input(
            "Tous les côtés sont-ils égaux ? (1 pour Oui, 0 pour Non) : "
        )
        if user_input_all_sides_equal.lower() == "exit":
            print("Au revoir!")
            break

        try:
            all_sides_equal = int(user_input_all_sides_equal)
            if all_sides_equal not in [0, 1]:
                print("Erreur : Veuillez entrer 1 (Oui) ou 0 (Non).")
                continue
        except ValueError:
            print("Erreur : Veuillez entrer un nombre valide.")
            continue

        # Prédiction
        result = predict_shape(n_sides, n_right_angles, all_sides_equal)
        print(f"La forme est : {result} !!!!!!!!!!!!!!!!!!!!")


# chatbox()
