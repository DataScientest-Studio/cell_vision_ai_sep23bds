# ProjectTemplate

## Presentation et Installation

Ce dépôt contient les codes pour le projet **BLOOD CELLS CLASSIFICATION**, développé durant le parcours [Data Scientist training](https://datascientest.com/en/data-scientist-course) de [DataScientest](https://datascientest.com/).

Le but de ce projet est de reconnaître et classifier des cellules du sang grâce à des algorithmes de machine learning et de deep learning. Les modèles développés pourront également aider à faciliter le diagnostic de la leucémie par la reconnaissance de blastes (cellules immatures fortement présentes chez les malades).

Structure du dépôt :
```bash
├── data (contient les dataframes répertoriant les images traîtées et leurs propriétés)
├── images (servant au streamlit
├── model (modèles de ML utilisés)
├── notebooks
│   ├── DataViz
│   └── ML
│   └── DL
└── streamlit
```

Le projet a été développé par :

- Wilfried Condemine ([GitHub]((https://github.com/wilfried1ier) / [LinkedIn](https://www.linkedin.com/in/wilfried-condemine-85065a54/))
- Michael Deroche ([GitHub](https://github.com/miklderoche) / [LinkedIn](https://www.linkedin.com/in/michaelderoche/))
- Claudia Mattei  ([GitHub](https://github.com/Claudia-Mattei) / [LinkedIn](www.linkedin.com/in/claudia-mattei))
- Charles Sallard ([GitHub](https://github.com/CharlesSALLARD) / [LinkedIn](https://www.linkedin.com/in/charles-sallard/))

Il faut installer les dépendances suivantes :

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
