# ProjectTemplate

## Presentation et Installation

Ce dépôt contient les codes pour le projet **BLOOD CELLS CLASSIFICATION**, développé durant le parcours [Data Scientist training](https://datascientest.com/en/data-scientist-course) de [DataScientest](https://datascientest.com/).

Le but de ce projet est de reconnaître et classifier des cellules du sang grâce à des algorithmes de machine learning et de deep learning. Les modèles développés pourront également aider à faciliter le diagnostic de la leucémie par la reconnaissance de blastes (cellules immatures fortement présentes chez les malades).

Structure du dépôt :
```bash
├── data (contient les dataframes répertoriant les images traîtées et leurs informations)
├── images (images utiles au streamlit)
├── model
│   ├── ML
│   └── DL
├── notebooks
│   ├── DataViz
│   └── ML
│   └── DL
└── app.py (application streamlit)
```
Les datasets utilisés sont présents aux adresses suivantes :

- [Dataset PBC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)
- [Dataset APL](https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl)
- [Dataset ALL_IDB](https://www.kaggle.com/nikhilsharma00/leukemia-dataset)

Le projet a été développé par :

- Wilfried Condemine ([GitHub](https://github.com/wilfried1ier) / [LinkedIn](https://www.linkedin.com/in/wilfried-condemine-85065a54/))
- Michael Deroche ([GitHub](https://github.com/miklderoche) / [LinkedIn](https://www.linkedin.com/in/michaelderoche/))
- Claudia Mattei  ([GitHub](https://github.com/Claudia-Mattei) / [LinkedIn](www.linkedin.com/in/claudia-mattei))
- Charles Sallard ([GitHub](https://github.com/CharlesSALLARD) / [LinkedIn](https://www.linkedin.com/in/charles-sallard/))

Il faut installer les dépendances suivantes :

```
pip install -r requirements.txt
```

## Streamlit App

L'application app.py peut être déployée ainsi via Streamlit Sharing :
- Allez sur [https://www.streamlit.io/sharing] et connectez-vous ou créez un compte Streamlit Sharing.
- Assurez-vous que votre application Streamlit est sur GitHub. Si ce n'est pas le cas, poussez votre application sur GitHub.
- Sur la page Streamlit Sharing, cliquez sur "New app" et suivez les instructions.
- Sélectionnez le référentiel que vous souhaitez déployer.
- Spécifiez le chemin vers votre application Streamlit et le fichier principal à exécuter.
- Appuyez sur le bouton "Deploy" pour lancer le déploiement de votre application.
- Votre application est maitenant en ligne (utiliser l'URL que vous avez spécifiée).

L'URL de notre application est la suivante : [https://ai-cellvision.streamlit.app/]
