import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# Initialisation de l'état de la session
if 'page' not in st.session_state:
    st.session_state['page'] = 'Accueil'

## %%% BARE LATERALE %%% ##

# Afficher l'image
st.sidebar.image('images/logo_VisionCellAI.png', width=150)

# Menu latéral avec des boutons
if st.sidebar.button('Accueil'):
    st.session_state['page'] = 'Accueil'
if st.sidebar.button('Analyse'):
    st.session_state['page'] = 'Analyse'
if st.sidebar.button('Démonstration'):
    st.session_state['page'] = 'Démonstration'
if st.sidebar.button('Résultats'):
    st.session_state['page'] = 'Résultats'
if st.sidebar.button('Documentation'):
    st.session_state['page'] = 'Documentation'

## %%% PAGE ACCUEIL %%% ##

# Affichage de la page en fonction de l'état de la session
if st.session_state['page'] == 'Accueil':

    # Créer une colonne pour centrer l'image
    col1, col2, col3 = st.columns([1, 2, 1])

    # Afficher l'image centrée dans la colonne du milieu (col2)
    with col2:
        st.image('images/illustration_accueil.png')
        
    st.title('Cell Vision AI')
    
    st.write(
    '''
    Dans le domaine de la médecine et de la recherche biomédicale, l'analyse des cellules dans les frottis sanguins 
    est d'une importance cruciale pour le diagnostic et la compréhension de nombreuses pathologies. Cependant, leur 
    analyse manuelle est fastidieuse, sujette à des erreurs humaines, et peut être chronophage. De plus, cela 
    nécessite de nombreux appareils relativement onéreux et l’utilisation de consommables potentiellement dangereux.
    '''
    )

    st.header('Objectif du Projet')
    st.write(
    '''
    L’objectif principal du projet **CellVisionAI** est la création d’algorithmes d’apprentissage machine et d’apprentissage profond 
    dédiés à la reconnaissance et la classification de cellules du sang. Cet outil pourrait être utilisé pour 
    faciliter le diagnostic de la leucémie en détectant des leucocytes anormaux.
    '''
    )

    st.write(
    '''
    **Projet réalisé par Wilfried Condemine, Michael Deroche, Claudia Mattei et Charles Sallard**.
    (Promotion sept23_DS Datascientest)
    '''
    )

## %%% PAGE PROJET %%% ##

elif st.session_state['page'] == 'Analyse':
    st.title("Analyse des jeux de données")

    # Charger le fichier CSV dans un DataFrame
    chemin_fichier_csv = "data/data_PBC.csv"
    df_data_PBC = pd.read_csv(chemin_fichier_csv)

    st.image('images/bandeau_analyse_5.jpg')

    tab1, tab2, tab3, tab4 = st.tabs(["PBC Dataset Normal DIB", "Leukemia Dataset", "Acute Promyelocytic Leukemia (APL)", "Recommandations"])

    with tab1:
        st.header("PBC Dataset Normal DIB")
    
        st.write(
            '''
            Le dataset contient des images de cellules sanguines normales d'individus sains, servant de base de référence 
            pour l'entraînement des modèles pour reconnaître différents types de cellules sanguines normales.
            
            **Description**

            - Type : Images JPEG
            - Volume : 268 Mo
            - Nombre d'Images : 17 092
            - Classes : 8
            
            Ce jeu de données contient des images de cellules normales individuelles, classées en huit catégories. Les images ont été acquises à l'Hôpital de Barcelone.
            '''
            )
        
        # Afficher df_data_PBC
        st.write(
            '''
            Afin de faciliter l'analyse, un dataset a été créé à partir des différentes informations disponibles à partir des images. 
            
            **data_PBC.csv :**
            '''
            )
        st.write(df_data_PBC)
        
        # Définir le texte avec une couleur de fond transparente
        texte_formatte = """
        <div style="background-color: #F0F0F5; padding: 20px; border-radius: 0px;">
        <p><strong>Analyse</strong></p>
        <p>            
        - Diversité des formes et des tailles des cellules dans les images.<br>
        - Diversité de propriétés entre les classes nous permettra de pouvoir classer les cellules.<br>
        - Le fond des images (basé sur la taille des hématies) semble indiquer que le zoom utilisé pour capturer les cellules est le même.<BR>
        - Distribution équilibrée des images entre les classes semblent adaptées à une utilisation dans des tâches d'analyse ou de modélisation.<BR>
        - Nécessité de normaliser la luminosité, la teinte et la taille des images.
        </p>
        </div>
        """
        
        # Afficher le texte formaté avec le fond transparent
        st.markdown(texte_formatte, unsafe_allow_html=True)
        
###@@@ GRAPHIQUES @@@###
        st.write('')
                
        # Grouper par type de cellule et compter le nombre d'images
        data = df_data_PBC['Classe'].value_counts().reset_index()
        data.columns = ['Type de cellule', 'Nombre d\'images']
        
        # Couleurs personnalisées pour les classes
        colors = ['#5f74f4', '#de5e45', '#57c89a', '#a16cf0', '#f7a460', '#5dcdf2', '#ee7193', '#c1e58d']

        # Créer une palette de couleurs personnalisée pour vos classes
        palette_couleurs = {
            'neutrophil': '#5f74f4',
            'eosinophil': '#de5e45',
            'ig': '#57c89a',
            'platelet': '#a16cf0',
            'erythroblast': '#f7a460',
            'monocyte': '#5dcdf2',
            'basophil': '#ee7193',
            'lymphocyte': '#c1e58d'
        }

## GRAPHIQUE BARRES #
        # Créer un graphique à BARRES avec plotly.graph_objects
        fig_bar = go.Figure(data=[go.Bar(
            x=data['Type de cellule'], 
            y=data['Nombre d\'images'], 
            text=data['Nombre d\'images'], 
            marker_color=colors[:len(data)],  # Applique les couleurs aux barres
            textposition='inside',
        )])

        # Mettre à jour la mise en page pour ajuster la taille et mettre un fond transparent
        fig_bar.update_layout(
            width=450,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': 'Distribution des types de cellules',
                'y':0.85,  # Ajustez la position en y si nécessaire
                'x':0.5,  # Ajustez la position en x si nécessaire
                'xanchor': 'center', 
                'yanchor': 'top',
                'font': {
                    'size': 15  # Ajustez la taille de la police comme nécessaire
                }
            },
            xaxis_title='Type de cellule',
            yaxis_title='Nombre d\'images'
        )
        
## GRAPHIQUE CAMEMBERT ##
        # Créer un graphique en camembert avec plotly.graph_objects
        fig_pie = go.Figure(data=[go.Pie(
            labels=data['Type de cellule'], 
            values=data['Nombre d\'images'],
            marker_colors=colors[:len(data)],  # Applique les couleurs aux segments
            textinfo='percent+label'
        )])

        # Mettre à jour la mise en page pour ajuster la taille et mettre un fond transparent
        fig_pie.update_layout(
            width=450,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': 'Proportions des types de cellules',
                'y':0.85,  # Ajustez la position en y si nécessaire
                'x':0.5,  # Ajustez la position en x si nécessaire
                'xanchor': 'center', 
                'yanchor': 'top',
                'font': {
                    'size': 15  # Ajustez la taille de la police comme nécessaire
                }
            },
            showlegend=False  # Ne pas afficher la légende
        )
        
        # Créer des colonnes pour afficher les graphiques côte à côte
        col1, col2 = st.columns(2)

        # Afficher le graphique à barres dans la première colonne
        with col1:
            st.plotly_chart(fig_bar)

        # Afficher le graphique en camembert dans la deuxième colonne
        with col2:
            st.plotly_chart(fig_pie)

        st.write("Echantillon d'images par type de cellules :")
        st.image('images/PBC_images.png')

##@@ GRAPHIQUE DES DIMENSIONS PAR CLASSE (HAUTEUR) @@##
        import plotly.express as px
        import plotly.subplots as sp   
        
        # Utiliser le DataFrame existant df_data_PBC
        df_graph_dim_class = df_data_PBC
        
        # Créer un dataframe avec les dimensions et les classes
        df_graph_dim_class[['Largeur', 'Hauteur']] = df_graph_dim_class['Dimensions'].str.split('x', expand=True)
        
        # Créer un graphique d'histogramme pour les largeurs
        fig_dimensions_largeur = px.histogram(df_graph_dim_class, x="Largeur", color="Classe",
                                              labels={"Largeur": "Largeur des images"},
                                              color_discrete_map=palette_couleurs,
                                              title="Répartition des largeurs des images")
        
        # Mettre à jour la mise en page pour ajuster la taille et mettre un fond transparent
        fig_dimensions_largeur.update_layout(
            width=325,  # Ajustez la largeur
            height=400,  # Ajustez la hauteur
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': 'Répartition des largeurs des images',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 15
                }
            },
            showlegend=False  # Ne pas afficher la légende
        )
        
        # Créer un graphique d'histogramme pour les hauteurs
        fig_dimensions_hauteur = px.histogram(df_graph_dim_class, x="Hauteur", color="Classe",
                                              labels={"Hauteur": "Hauteur des images"},
                                              color_discrete_map=palette_couleurs,
                                              title="Répartition des hauteurs des images")
        
        # Mettre à jour la mise en page pour ajuster la taille et mettre un fond transparent
        fig_dimensions_hauteur.update_layout(
            width=400,  # Ajustez la largeur
            height=400,  # Ajustez la hauteur
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': 'Répartition des hauteurs des images',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 15
                }
            },
            showlegend=False  # Ne pas afficher la légende
        )
        # Utiliser st.beta_columns pour afficher les graphiques côte à côte
        col1, col2 = st.columns(2)
        
        # Afficher le graphique de la boîte à moustaches de la teinte dans la première colonne
        with col1:
            st.plotly_chart(fig_dimensions_largeur)
        
        # Afficher le graphique de la boîte à moustaches de la luminosité dans la deuxième colonne
        with col2:
            st.plotly_chart(fig_dimensions_hauteur)

##@@ BOÎTES À MOUSTACHES DE LA TEINTE ET DE LA LUMINOSITÉ PAR CLASSE @@##
        
        # Créer une figure pour la boîte à moustaches de la teinte par classe
        fig_hue_box = px.box(df_data_PBC, x='Classe', y='Teinte', color='Classe',
                             color_discrete_map=palette_couleurs,
                             title="Boîtes à moustaches de la Teinte")
        
        # Mettre à jour la mise en page pour ajuster la taille et mettre un fond transparent
        fig_hue_box.update_layout(
            width=800,  # Ajustez la valeur comme nécessaire
            height=500,  # Ajustez la valeur comme nécessaire
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': "Boîtes à moustaches de la Teinte",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 15
                }
            },
            showlegend=False  # Ne pas afficher la légende
        )
        st.plotly_chart(fig_hue_box)
        
        # Créer une figure pour la boîte à moustaches de la luminosité par classe
        fig_brightness_box = px.box(df_data_PBC, x='Classe', y='Luminosité', color='Classe',
                                   color_discrete_map=palette_couleurs,
                                   title="Boîtes à moustaches de la Luminosité")
        
        # Mettre à jour la mise en page pour ajuster la taille et mettre un fond transparent
        fig_brightness_box.update_layout(
            width=800,  # Ajustez la valeur comme nécessaire
            height=500,  # Ajustez la valeur comme nécessaire
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': "Boîtes à moustaches de la Luminosité",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 15
                }
            },
            showlegend=False  # Ne pas afficher la légende
        )
        st.plotly_chart(fig_brightness_box)
            
    with tab2:
        st.header("Leukemia Dataset")

        # Charger le jeu de données depuis le fichier CSV
        chemin_fichier_csv = "data/data_leukemia_dataset.csv"
        df_data_leukemia_dataset = pd.read_csv(chemin_fichier_csv)
        
        st.write(
            '''
             Le dataset contient des images de cellules sanguines de patients sains et de patients atteints de Leucémie Lymphoblastique Aiguë (ALL), avec des informations sur les centroïdes. 
             Il est utile pour tester la segmentation, la classification, et les méthodes de prétraitement des images.
             
            **Description**
            
            - Type : Images JPEG et TIFF
            - Sous-ensembles : ALL_IDB1 (143 Mo, 108 images), ALL_IDB2 (49 Mo, 260 images)
            - Classes : Patients sains et patients atteints de Leucémie Lymphoblastique Aiguë (ALL)
            
            Ce jeu de données contient des images de cellules sanguines provenant de patients sains et atteints de leucémie.
            '''
            )

        # Afficher les 5 premières lignes de df_data_leukemia_dataset
        st.write(
            '''
            Afin de faciliter l'analyse, un dataset a été créé à partir des différentes informations des images dans les dossiers ALL_IDB1 et ALL_IDB2. 
            
            **data_leukemia_dataset.csv :**
            '''
            )
        st.write(df_data_leukemia_dataset)
        
        # Définir le texte avec une couleur de fond transparente
        texte_formatte = """
        <div style="background-color: #F0F0F5; padding: 20px; border-radius: 0px;">
        <p><strong>Analyse</strong></p>
        <p>            
        - Une limitation est l'absence de classification des cellules, rendant difficile la vérification de la performance des modèles.<br>
        - Les coordonnées des centroïdes des cellules sont fournies pour les images ALL_IDB1.<br>
        - Diversité des données pour construire un modèle robuste.
        </p>
        </div>
        """
        
        # Afficher le texte formaté avec le fond transparent
        st.markdown(texte_formatte, unsafe_allow_html=True)

###@@@ GRAPHIQUES @@@###

        # Créer une nouvelle colonne pour différencier les patients sains et malades
        df_data_leukemia_dataset['Statut patient'] = df_data_leukemia_dataset['Leucémie_ALL'].apply(lambda x: 'Malade' if x == 1 else 'Sain')

        st.write('')
        
        # Afficher la distribution des classes en distinguant les patients sains et malades
        fig1 = px.histogram(df_data_leukemia_dataset, x='Classe', color='Statut patient')
        fig1.update_layout(title={'text': 'Distribution des classes ALL_IDB1 et ALL_IDB2 en distinguant les patients sains et malades', 'x':0.5, 'xanchor': 'center'})
        st.plotly_chart(fig1)
        
        # Afficher la distribution de la dimension pour les classes ALL_IDB1 et ALL_IDB2
        fig2 = px.histogram(df_data_leukemia_dataset, x='Dimensions', color='Classe')
        fig2.update_layout(title={'text': 'Distribution de la dimension pour les classes ALL_IDB1 et ALL_IDB2', 'x':0.5, 'xanchor': 'center'})
        st.plotly_chart(fig2)
        
        # Afficher la distribution de la résolution pour les classes ALL_IDB1 et ALL_IDB2
        fig3 = px.histogram(df_data_leukemia_dataset, x='Résolution', color='Classe')
        fig3.update_layout(title={'text': 'Distribution de la résolution pour les classes ALL_IDB1 et ALL_IDB2', 'x':0.5, 'xanchor': 'center'})
        st.plotly_chart(fig3)
        
        # Ajouter une boîte à moustaches pour la luminosité
        fig4 = px.box(df_data_leukemia_dataset, x='Classe', y='Luminosité', color='Classe')
        fig4.update_layout(title={'text': 'Distribution de la luminosité pour les classes ALL_IDB1 et ALL_IDB2', 'x':0.5, 'xanchor': 'center'})
        st.plotly_chart(fig4)
        
        # Ajouter une boîte à moustaches pour la teinte
        fig5 = px.box(df_data_leukemia_dataset, x='Classe', y='Teinte', color='Classe')
        fig5.update_layout(title={'text': 'Distribution de la teinte pour les classes ALL_IDB1 et ALL_IDB2', 'x':0.5, 'xanchor': 'center'})
        st.plotly_chart(fig5)

    with tab3:
        st.header("Acute Promyelocytic Leukemia (APL)")
        
        # Charger le fichier CSV dans un DataFrame
        chemin_fichier_csv = "data/data_APL.csv"
        df_data_APL = pd.read_csv(chemin_fichier_csv, sep=';', error_bad_lines=False)
        
        st.write(
            '''
            Le dataset est composé d'images de cellules sanguines classées de patients atteints de différentes formes de leucémie. 
            Certaines cellules n'ont pas été classées, introduisant de l'incertitude dans la variable cible.
            
            **Description**
            
            - Type : Images JPEG
            - Volume : 515 Mo
            - Nombre d'Images : 25 721
            - Origine : Cellules de 106 patients de l’Hôpital Johns Hopkins, atteints de Leucémie Myéloïde Aiguë (AML) ou de Leucémie Aiguë Promyélocytaire (APL)
            
            Ce jeu de données contient des images de cellules de patients atteints de deux types de leucémie. Les cellules sont classées dans des dossiers par catégorie, et il existe également un dossier pour les cellules non classées.
            '''
            )
        
            # Afficher df_data_APL
        st.write(
            '''
            Afin de faciliter l'analyse, un dataset a été créé à partir des différentes informations disponibles à partir des images. 
            
            **data_APL.csv :**
            '''
            )
        st.write(df_data_APL)
        
        st.write(
                '''
            **Analyse**
            
            - **Conditions d'Acquisition :** 
            Les conditions d'acquisition des images semblent similaires à celles du Dataset 1.
            - **Classification des Cellules :**
            Plus de 15 000 images sont classées selon le type de cellules, mais environ 10 000 images ne sont pas classées.
            - **Artifacts :**
            Certains types de cellules, tels que les "smudge cells", contiennent de nombreux outliers et pourraient ne pas être utiles pour l'analyse.
            - **Variations de Taille :**
            Des variations de taille plus importantes sont observées par rapport au Dataset 1, sans dépendance apparente avec les classes de cellules.
            - **Informations des Patients :**
            Un fichier master.csv contient les diagnostics et quelques informations sur les patients. Il y a une répartition équilibrée des données entre les sexes et les tranches d'âge, 
            avec une prédominance masculine conforme à la prévalence de la maladie.
            '''
            )
        
    with tab4:
        st.header("Recommandations pour le traitement des données")

        st.write(
            '''
            - **Analyse Exploratoire des Données :**
            Effectuer une analyse approfondie pour comprendre les limites potentielles et les caractéristiques spécifiques de chaque jeu de données.
            
            - **Nettoyage des Données :**
            Pour le Dataset 3, bien que la majorité des données soient propres, une analyse détaillée des images peut révéler des problèmes, 
            tels que la difficulté de reconnaissance des noyaux par les algorithmes, nécessitant l'élimination de certaines données.
            
            - **Prétraitement :**
            Des fichiers de données au format CSV ont été créés pour simplifier l’analyse statistique des données, y compris le nombre de classes, 
            le nombre d’images par classe, et les formats et dimensions des images.
            
            - **Gestion des Limitations :**
            Les différences en termes de qualité, de format, et de représentativité des catégories de cellules entre les jeux de données peuvent poser des défis.
            Une stratégie d'intégration et de normalisation doit être développée pour gérer ces différences.
            Le manque de classification des cellules dans le Dataset 2 et les incertitudes dans le Dataset 3 nécessitent des stratégies spécifiques pour gérer et, si possible, minimiser ces limitations.
            '''
            )
## %%% PAGE DEMONSTRATION %%% ##

elif st.session_state['page'] == 'Démonstration':
    st.title("Démonstration")
    st.header('Objectif de la démonstration')

    tab5, tab6 = st.tabs(["Machine Learning", "Deep Learning"])

    with tab5:
        st.write(
        '''
        La démonstration du projet **CellVisionAI** consiste en une interface utilisateur permettant à l'utilisateur de charger une image de frottis sanguin. L'outil est ensuite capable de reconnaître et de classifier les cellules du sang présentes dans l'image.
        '''
    )
    
    with tab6:
        st.write(
        '''
        La démonstration du projet **CellVisionAI** consiste en une interface utilisateur permettant à l'utilisateur de charger une image de frottis sanguin. L'outil est ensuite capable de reconnaître et de classifier les cellules du sang présentes dans l'image.
        '''
    )

## %%% PAGE RESULTATS %%% ##

elif st.session_state['page'] == 'Résultats':
    st.title("Résultats")
    st.header('Objectif des résultats')

    st.write(
        '''
        Les résultats du projet **CellVisionAI** sont encore en cours d'évaluation. Cependant, les premiers résultats sont prometteurs. L'outil est capable de reconnaître et de classifier avec précision les cellules du sang présentes dans les frottis sanguins.
        '''
    )

## %%% PAGE DOCUMENTATION %%% ##

elif st.session_state['page'] == 'Documentation':
    st.title("Documentation")
    
    tab7, tab8, tab9 = st.tabs(["Datasets", "Bibliographie", "Crédits"])

    with tab7:
        st.write(
            '''
            - [PBC Dataset Normal DIB - National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)
            - [Acute Promyelocytic Leukemia (APL) - Kaggle](https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl)
            - [Leukemia Dataset - Kaggle](https://www.kaggle.com/nikhilsharma00/leukemia-dataset)
        '''
        )
    
    with tab8:
        st.write(
        '''
        - [Recognition of peripheral blood cell images using convolutional neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub)
        - [A deep learning model (ALNet) for the diagnosis of acute leukaemia lineage using peripheral blood cell images](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000742?via%3Dihub)
        '''
        )
        
    with tab9:
        st.write(
        '''
        Le logo CellVisionAI et les images d'illustrations ont été générées par DALL•E 3.
        '''
        )
