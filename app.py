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
if st.sidebar.button('Analyse des jeux de données'):
    st.session_state['page'] = 'Analyse des jeux de données'
if st.sidebar.button('Machine learning'):
    st.session_state['page'] = 'Machine learning'
if st.sidebar.button('Deep learning'):
    st.session_state['page'] = 'Deep learning'
if st.sidebar.button('Transfer learning'):
    st.session_state['page'] = 'Transfer learning'
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

elif st.session_state['page'] == 'Analyse des jeux de données':
    st.title("Analyse des jeux de données")

    st.image('images/bandeau_analyse_5.jpg')

    tab1, tab2, tab3, tab4 = st.tabs(["PBC Dataset Normal DIB", "Leukemia Dataset", "Acute Promyelocytic Leukemia (APL)", "Nos recommandations"])

    
######################################
      # PBC Dataset Normal DIB #
######################################   
    
    # Charger le fichier CSV dans un DataFrame
    chemin_fichier_csv = "data/data_PBC_1.csv"
    df_data_PBC = pd.read_csv(chemin_fichier_csv)
    
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

# Echantillon d'image #
        
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

    
######################################
        # Leukemia Dataset #
######################################      

    # Charger le jeu de données depuis le fichier CSV
    chemin_fichier_csv = "data/data_leukemia_dataset.csv"
    df_data_leukemia_dataset = pd.read_csv(chemin_fichier_csv) 

    with tab2:
        st.header("Leukemia Dataset")
        
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

    
######################################
# Acute Promyelocytic Leukemia (APL) #
######################################
    
    # Charger le fichier CSV dans un DataFrame
    chemin_fichier_apl_csv = "data/data_APL_streamlit_4.csv"
    df_data_APL = pd.read_csv(chemin_fichier_apl_csv)

    with tab3:
        st.header("Acute Promyelocytic Leukemia (APL)")
        
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

        # Définir le texte avec une couleur de fond transparente
        texte_formatte = """
        <div style="background-color: #F0F0F5; padding: 20px; border-radius: 0px;">
        <p><strong>Analyse</strong></p>
        <p>            
        - Les conditions d'acquisition des images semblent similaires à celles du Dataset 1.<br>
        - Plus de 15 000 images sont classées (Signed slides) selon le type de cellules, mais environ 10 000 images (Unsigned slides) ne sont pas classées.<br>
        - Certains types de cellules, tels que les "smudge cells", contiennent de nombreux outliers et pourraient ne pas être utiles pour l'analyse.<br>
        - Des variations de taille plus importantes sont observées par rapport au Dataset 1, sans dépendance apparente avec les classes de cellules.<br>
        - Un fichier master.csv contient les diagnostics et quelques informations sur les patients. Il y a une répartition équilibrée des données entre les sexes et les tranches d'âge,avec une prédominance masculine conforme à la prévalence de la maladie.
        </p>
        </div>
        """
        
        # Afficher le texte formaté avec le fond transparent
        st.markdown(texte_formatte, unsafe_allow_html=True)

###@@@ GRAPHIQUES @@@###

# Répartition des classes
        
        # Charger les données
        df_data_APL = pd.read_csv("data/data_APL_streamlit_4.csv") 
        
        # Créer un graphique de répartition des classes
        fig = px.histogram(df_data_APL, x="Classe", title="Répartition des classes")
        
        # Personnalisation du graphique
        fig.update_traces(marker=dict(color=px.colors.qualitative.Set3))  # Couleurs différentes pour chaque classe
        fig.update_xaxes(categoryorder="total descending")  # Tri des classes par ordre décroissant
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=600)

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

# Echantillon d'images par type de cellules

        st.write("Echantillon d'images par type de cellules :")
        st.image('images/APL_echantillon.png')
        
# Répartition des classes du Patient_00 au Patient_105
        
        # Extraire les numéros de patients pour le tri
        df_data_APL['Patient_number'] = df_data_APL['Nom du patient'].str.extract('(\d+)').astype(int)
        
        # Créer un graphique à barres empilées vertical en utilisant la fonction count()
        fig = px.bar(df_data_APL.groupby(['Nom du patient', 'Classe']).size().reset_index(name='Nombre d\'images'), 
                     x="Nom du patient", y="Nombre d'images", color="Classe",
                     title="Répartition des classes du Patient_00 au Patient_105", orientation='v')
        
        # Personnaliser le graphique
        fig.update_layout(xaxis_title="Nom du patient", yaxis_title="Nombre d'images")
        fig.update_traces(marker=dict(line=dict(width=0)))  # Supprimer les contours des barres
        
        # Trier les patients en utilisant le numéro de patient
        noms_des_patients = df_data_APL['Nom du patient'].unique().tolist()
        noms_des_patients.sort(key=lambda x: int(x.split('_')[1]))  # Tri par numéro de patient
        fig.update_xaxes(categoryorder="array", categoryarray=noms_des_patients)
        fig.update_layout(height=650, width=1000)
        
        # Faire pivoter les légendes de l'axe x à 45 degrés
        fig.update_xaxes(tickangle=45)
        
        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

# Distribution des dimensions des images par classe
        # Créer un graphique d'histogramme de la dimension des images par classe
        fig = px.histogram(df_data_APL, x="Dimension", color="Classe", title="Distribution des dimensions des images par classe")
        
        # Personnaliser le graphique
        fig.update_layout(xaxis_title="Dimension de l'image", yaxis_title="Nombre d'images")
        fig.update_xaxes(type="category", tickangle=45)  # Utiliser une échelle catégorielle pour la dimension
        fig.update_traces(marker=dict(line=dict(width=0)))  # Supprimer les contours des barres
        fig.update_layout(height=700, width=900)

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

# Distribution des classes par luminosité et teinte

        # Créer un graphique de nuage de points 2D (scatter plot)
        fig = px.scatter(df_data_APL, x="Luminosité", y="Teinte", color="Classe",
                         title="Distribution des classes par luminosité et teinte")
        
        # Personnaliser le graphique
        fig.update_layout(xaxis_title="Luminosité", yaxis_title="Teinte")
        fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=700, width=900)
        # Afficher le graphique
        st.plotly_chart(fig)

# box plot pour la distribution des classes par luminosité et la teinte

        # Définir une séquence de couleurs personnalisée
        couleurs_classes = px.colors.qualitative.Plotly[:len(df_data_APL['Classe'].unique())]
        
        # Créer un box plot pour la distribution des classes par luminosité
        fig_luminosite = px.box(df_data_APL, x="Classe", y="Luminosité", color="Classe", title="Distribution des classes par luminosité",
                                color_discrete_sequence=couleurs_classes)
        
        # Personnaliser le graphique
        fig_luminosite.update_layout(yaxis_title="Luminosité")
        fig_luminosite.update_xaxes(tickangle=45)
        fig_luminosite.update_layout(height=800, width=900)
        
        # Afficher le graphique de luminosité
        st.plotly_chart(fig_luminosite)
        
        # Créer un box plot pour la distribution des classes par teinte
        fig_teinte = px.box(df_data_APL, x="Classe", y="Teinte", color="Classe", title="Distribution des classes par teinte",
                            color_discrete_sequence=couleurs_classes)
        
        # Personnaliser le graphique
        fig_teinte.update_layout(yaxis_title="Teinte")
        fig_teinte.update_xaxes(tickangle=45)
        fig_teinte.update_layout(height=800, width=900)
        
        # Afficher le graphique de teinte
        st.plotly_chart(fig_teinte)

        
######################################
       # Nos Recommandations #
######################################

    
    with tab4:
        st.header("Nos recommandations")

        st.write(
            '''
            Le **dataset 1** contient des images de cellules sanguines normales provenant d'individus sains, 
            ce qui en fait une base de données de référence pour entraîner et tester des modèles d'apprentissage automatique 
            et d'apprentissage profond afin de reconnaître différents types de cellules sanguines normales.

            Le **dataset 2** propose des images de cellules sanguines de patients atteints de ALL, avec des informations sur les centroïdes. 
            Il peut être utilisé pour tester la capacité de segmentation des algorithmes, ainsi que les méthodes de prétraitement des images. 
            On pourrait penser à une classification des images en cellules sanguines normales et cellules anormales de patients atteints de Leucémie Lymphoblastique Aiguë (ALL).
            
            Le **dataset 3** est composé d'images de cellules sanguines classées de patients atteints de différentes formes de leucémie (AML ou APL). 
            L'apprentissage en profondeur permettrait de différencier les cellules et diagnostiquer l'APL à partir de la morphologie cellulaire. 
            
            **Limitations par les données :**
            - Qualité des données : biais engendrés par des erreurs, artefacts, défauts dans les images ou la présence de plusieurs cellules sur la même image.
            - Taille des données : nombre limité d'images dans ALL_IDB2 du dataset 2.
            - Manque de classement : pas de classification des cellules dans le dataset 2 ce qui rendra impossible la vérification de la performance de notre modèle sur cet ensemble de données. 
            De même pour une partie du dataset 3 (‘Unsigned slides’).
            - Différences entre les jeux de données : en termes de qualité, de format, de zoom sur les cellules, de conditions d’acquisition des images ou de représentativité des catégories de cellules.
            '''
            )

        # Définir le texte avec une couleur de fond transparente
        texte_formatte = """
        <div style="background-color: #F0F0F5; padding: 20px; border-radius: 0px;">
        <p><strong>Choix</strong></p>
        <p>            
        - Utiliser principalement le dataset 1 (PBC Dataset Normal DIB) pour un modèle de classification des cellules sanguines normales. <br>
        - Le dataset 3 (Acute Promyelocytic Leukemia (APL)) viendra compléter cette base de données avec les cellules ‘Signed’ et permettrait éventuellement d’identifier des cellules sanguines anormales, 
        caractéristiques de patients atteints de leucémie.<br>
        - Exclusion du dataset 2 (Leukemia Dataset).<br>
        - Mettre en place des techniques de prétraitement d'images, de segmentation, d'apprentissage machine et d’apprentissage profond 
        pour extraire des caractéristiques pertinentes à partir des images.<br>
        </p>
        </div>
        """
    
        # Afficher le texte formaté avec le fond transparent
        st.markdown(texte_formatte, unsafe_allow_html=True)

## %%% PAGE MACHINE LEARNING %%% ##

elif st.session_state['page'] == 'Machine learning':
    st.title("Machine learning")

    tab5, tab6 = st.tabs(["Extraction des caractéristiques", "Modèles de Machine learning"])

    with tab5:
        st.write(
        '''
        l’enjeu de l’extraction des données en vue de leur utilisation pour un modèle de Machine Learning réside dans l’automatisation du processus pour chacun des datasets identifiés.
        
        Après de multiples itérations, et l’utilisation de différentes bibliothèques d’analyses d’images (OpenCV et skimage), 
        nous avons établi deux méthodologies qui, malgré leurs imperfections, sont suffisamment solides pour répondre aux besoins de notre modèle de Machine Learning. 
        
        Nous présentons ici la 2e méthode qui a extrait les caractéristiques les plus optimisées :
        '''
    )
        
        # 1ère étape
        st.subheader("Préparation des données et segmentation initiale")
        st.markdown("""
        - Importation des bibliothèques nécessaires au traitement d'images, à la gestion de données et à la visualisation.
        - Extraction des chemins des images pour simplifier l'accès aux données.
        - Définition des classes cibles pour le dataset, en se concentrant sur les types de cellules sanguines d'intérêt.
        - Segmentation initiale des cellules basée sur l'espace couleur LAB, en utilisant le canal b pour distinguer les cellules du fond.
        - Utilisation de la méthode d'Otsu pour choisir automatiquement le seuil de segmentation.
        - Amélioration de la segmentation par des opérations morphologiques telles que la fermeture et l'ouverture.
        
        Visualisation des étapes de segmentation initiale :
        """)

        st.image('images/M2_6.jpg')
        st.image('images/M2_8.jpg')
        st.image('images/M2_9.jpg')
        st.image('images/M2_7.jpg')
        
        # 2ème étape
        st.subheader("Extraction des caractéristiques à partir des boîtes encadrantes")
        st.markdown("""
        - Identification des boîtes encadrantes des cellules segmentées.
        - Segmentation des noyaux des cellules d'intérêt à partir des boîtes encadrantes.
        - Extraction des caractéristiques quantitatives des noyaux, telles que l'aire, le périmètre, l'excentricité, etc.
        - Analyse individuelle de chaque cellule pour calculer ces caractéristiques.
        - Stockage des caractéristiques dans une structure de données optimale pour l'entraînement de modèles de machine learning.
        
        Visualisation des noyaux des cellules d'intérêt et des caractéristiques extraites à partir des boîtes encadrantes :
        """)

        st.image('images/M2_3.jpg')
        
        # 3ème étape
        st.subheader("Traitement global et stockage des caractéristiques")
        st.markdown("""
        - Traitement de l'ensemble des images de la base de données, avec extraction systématique des caractéristiques de chaque noyau.
        - Conservation des informations sur les régions des noyaux d'intérêt identifiées par la segmentation.
        - Enregistrement de toutes les caractéristiques extraites dans un fichier CSV pour chaque dataset.
        - Concaténation des données en un seul dataset pour la création du modèle de machine learning.
        
        Visualisation des données traitées et enregistrées :
        """)

        st.image('images/M2_4.jpg')

        # Liste des variables retenues
        st.subheader("Liste des variables retenues")
        st.markdown("""
        - **Nom :** nom de l'image initiale.
        - **Numéro :** si plusieurs cellules sur l'image, numéro de la cellule analysée.
        - **Aire noyau :** surface occupée par le noyau sur l'image en pixels.
        - **Périmètre noyau :** longueur du contour occupé par le noyau en pixels.
        - **Largeur du rectangle minimal :** largeur du rectangle minimal englobant le noyau.
        - **Hauteur du rectangle minimal :** hauteur du rectangle minimal englobant le noyau.
        - **Rayon noyau :** rayon du cercle minimal englobant le noyau.
        - **Petit axe noyau :** petit axe de l'ellipse minimale englobant le noyau.
        - **Grand axe noyau :** grand axe de l'ellipse minimale englobant le noyau.
        - **Excentricité noyau :** excentricité de l'ellipse minimale englobant le noyau.
        - **Périmètre convexe noyau :** périmètre de la forme convexe minimale englobant le noyau.
        - **Solidité noyau :** rapport de l'aire de l'aire du noyau sur l'aire de la forme convexe.
        - **Boîte encadrante :** coordonnées de la boîte encadrant le noyau.
        - **Centre :** 1 si la cellule recouvre le centre de l'image sinon 0 (plusieurs types cellulaires peuvent être présentes sur la même image, ce terme permet de repérer la cellule principale si c'est le cas).
        - **Classe :** type cellulaire.

        La variable cible est la “Classe” des cellules. 
        """)

    
        # Conclusion
        st.subheader("Conclusion")
        st.markdown("""
        - Suppression des images avec une mauvaise segmentation et des valeurs aberrantes.
        - Élimination des classes sous-représentées, des images non identifiées et des classes incompatibles.
        - Renommage ou regroupement de certaines classes pour harmoniser les datasets.
        - Enregistrement des données extraites dans un fichier CSV pour l'entraînement de modèles de machine learning.
        
        Ces étapes préparent les données de manière efficace pour une future classification des cellules sanguines.
        """)

        st.image('images/M2_5.jpg')
        
    with tab6:

        # Titre
        st.header("Analyse des Algorithmes de Classification")
        
        # Standardisation des données et séparation en ensembles d'entraînement et de test
        st.markdown("<h4>Première étape</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Standardisation des données pour préparer les algorithmes sensibles aux échelles.
        - Séparation des données en ensembles d'entraînement et de test.
        """)

        # Résultats de l'algorithme KNN
        st.markdown("<h4>K-Nearest Neighbors (KNN)</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Accuracy sur l'ensemble de test : 0,68.
        
        """)
        
        # Résultats de l'algorithme RandomForest
        st.markdown("<h4>RandomForest</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Accuracy sur l'ensemble de test : 0,70.
        - Rapport de classification pour chaque classe.
        
        Optimisation des hyperparamètres avec GridSearchCV, mais pas d'amélioration significative.
        L'observation montre que les classes sous-représentées sont mal prédites. Le sur-échantillonnage avec SMOTE ne fait que baisser les performances.
        
        Classes mal prédites : "erythroblast", "monocyte", "ig", "basophil".
        
        """)
        
        # Résultats de l'algorithme XGBoost
        st.markdown("<h4>XGBoost</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Accuracy sur l'ensemble de test : 0,71.
        
        """)
        
        # Résultats de l'algorithme SVM
        st.markdown("<h4>Support Vector Machine (SVM)</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Accuracy sur l'ensemble de test : 0,73 (après optimisation des hyperparamètres : C=100, gamma=0,01, kernel='rbf').
        - Pas de sur-apprentissage (accuracy sur l'ensemble d'entraînement : 0,72).
        
        Classes mal prédites : "erythroblast" confondue avec "lymphocyte", "monocyte" et "basophil" confondus avec "ig".
        
        """)
        
        # Résultats du réseau de neurones avec Keras
        st.markdown("<h4>Réseau de Neurones avec Keras</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Accuracy sur l'ensemble de test : 0,73.
        - Augmentation de l'accuracy en éliminant les cellules des 4 classes extrêmement sous-représentées.
        
        Meilleurs scores obtenus avec un réseau de neurones dense avec Keras. Architecture du modèle :
        - Couche Dense à 64 neurones, fonction d'activation 'relu'.
        - Couche Dense à 32 neurones, fonction d'activation 'relu'.
        - Couche Dense à 13 neurones, fonction d'activation 'softmax'.
        - Optimiseur : 'adam'.
        - Fonction de perte : 'sparse_categorical_crossentropy'.
        - Entraînement sur 100 époques avec un batch_size=128.
        
        Classes mal prédites : "erythroblast" confondue avec "lymphocyte", "monocyte" et "basophil" confondus avec "ig".
        
        """)

        # Texte à afficher
        text = """
                      precision    recall  f1-score   support
        
                   0       0.00      0.00      0.00        16
                   1       0.49      0.67      0.57       246
                   2       0.74      0.84      0.78       341
                   3       0.79      0.86      0.82       623
                   4       0.69      0.53      0.60       363
                   5       0.00      0.00      0.00         6
                   6       0.67      0.52      0.59       556
                   7       0.68      0.87      0.76       655
                   8       0.12      0.02      0.03        58
                   9       0.62      0.56      0.59       408
                  10       0.87      0.87      0.87       842
                  11       0.00      0.00      0.00         4
                  12       0.91      0.79      0.85       421
                  13       0.00      0.00      0.00         4
        
            accuracy                           0.73      4543
           macro avg       0.47      0.47      0.46      4543
        weighted avg       0.73      0.73      0.72      4543
        """
        st.code(text, language='text')

        st.image('images/ML_Keras.jpg')
        
        st.write("")
        
        # conclusion
        texte_formatte = """
        <div style="background-color: #F0F0F5; padding: 20px; border-radius: 0px;">
        <p><h4>Conclusion</h4></p>
        <p>            
        - Les performances de classification sont affectées par les classes sous-représentées et la ressemblance morphologique entre certaines cellules.<br>
        - L'algorithme de segmentation des noyaux se concentre sur la forme et la taille, ne prenant pas en compte d'autres caractéristiques importantes.<br>
        - Le meilleur score est atteint avec un réseau de neurones dense avec Keras après élimination des classes extrêmement sous-représentées.<br>
        - Il existe des limitations à l'obtention d'un score élevé en raison de la complexité des images cellulaires.
        </p>
        </div>
        """
        
        # Afficher le texte formaté avec le fond transparent
        st.markdown(texte_formatte, unsafe_allow_html=True)

## %%% PAGE DEEP LEARNING %%% ##

elif st.session_state['page'] == 'Deep learning':
    st.title("Deep learning")
    st.header('Objectif des résultats')

    st.write(
        '''
        Les résultats du projet **CellVisionAI** sont encore en cours d'évaluation. Cependant, les premiers résultats sont prometteurs. L'outil est capable de reconnaître et de classifier avec précision les cellules du sang présentes dans les frottis sanguins.
        '''
    )

## %%% PAGE DEEP LEARNING %%% ##

elif st.session_state['page'] == 'Transfer learning':
    st.title("Transfer learning")
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
