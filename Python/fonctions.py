import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import fnmatch
resultat_indiv={}
res_gradient_vib={}
dico_intens_vide={}
rapport_cyclique = np.linspace(440,1000,15)
moy_ec_intens={int(intens) : [] for intens in rapport_cyclique}
def compter_csv_commencant_par(dossier, prefixe):
    nombre_fichiers = 0
    for fichier in os.listdir(dossier):
        if fnmatch.fnmatch(fichier, f"{prefixe}*.csv"):
            nombre_fichiers += 1
    return nombre_fichiers
def ajout_dico_evol(prenom,numtestdep=1,numtestarr=0): #
    """
    Ajoute les test allant de numtestdep à numtestarr (inclus) de prenom au dico resultat_indiv
    """
    if numtestarr==0:
        numtestarr = compter_csv_commencant_par("D:/Kovelia/Documents/Cours/PREPA/A2/TIPE/CODE/DONNEES/SESSION TEST/ ".strip() + prenom.lower(),prenom.lower() +'test')
    resultat_indiv[prenom.lower()]=[0,[],[]] #Nom des individus : nb de tests, liste des temps de réac au buzzer, liste des temps au vibreur. Toute les 5 données on a un jour (Vibreur et buzzer sont les mêmes.
    for i in range(numtestdep,numtestarr+1,1):
        resultat_indiv[prenom.lower()][0]+=1
        with open ("D:/Kovelia/Documents/Cours/PREPA/A2/TIPE/CODE/DONNEES/SESSION TEST/ ".strip()+prenom.lower()+"/ ".strip() + prenom.lower() +'test'+str(i)+'.csv',newline='') as fichier_csv:
            lecteur= csv.reader(fichier_csv, delimiter=';')
            #l=lecteur.readlines()
            #print(l)
            compteur=0
            lecteur.__next__()
            for ligne in lecteur:
                if any(field.strip() for field in ligne):
                    try:
                        compteur+=1
                        if compteur<=5:
                            resultat_indiv[prenom.lower()][1].append(int(ligne[0]))
                        else:
                            resultat_indiv[prenom.lower()][2].append(int(ligne[0]))
                    except:
                        pass
def ajout_intens(prenom, numtestdep=1 ,numtestarr=0): #FIXME GROS PROBLEME SUR LE DICO FINAL A CHANGER AU PLUS VITE
    if numtestarr == 0:
        numtestarr = compter_csv_commencant_par("D:/Kovelia/Documents/Cours/PREPA/A2/TIPE/CODE/DONNEES/SESSION TEST/ ".strip() + prenom.lower(),prenom.lower() +"intens")
    dico_intens_vide = {int(i): [] for i in rapport_cyclique}
    res_gradient_vib[prenom.lower()] = dico_intens_vide.copy()  # Nom des individus : nb de tests, liste des temps de réac au buzzer, liste des temps au vibreur. Toute les 5 données on a un jour (Vibreur et buzzer sont les mêmes.
    for i in range(numtestdep, numtestarr + 1, 1):
        with open("D:/Kovelia/Documents/Cours/PREPA/A2/TIPE/CODE/DONNEES/SESSION TEST/ ".strip()+prenom.lower()+"/ ".strip() + prenom.lower() +'intens'+str(i)+'.csv',newline='') as fichier_csv:
            lecteur = csv.reader(fichier_csv, delimiter=';')

            next(lecteur)
            for ligne in lecteur:
                if any(field.strip() for field in ligne):
                    try:
                        res_gradient_vib[prenom.lower()][int(ligne[1])].append(int(ligne[0]))
                    except(KeyError, ValueError):
                        pass

def enregistrement_donnees():
    """
    Enregistre dans le fichier "resultat_indiv.csv" les données du dictionnaire
    :return:
    """
    with open("D:/Kovelia/Documents/Cours/PREPA/A2/TIPE/CODE/DONNEES/BASE DONNEES/resultat_indiv.csv", "w") as f:
        writer = csv.writer(f)
        for k, v in resultat_indiv.items():
            writer.writerow([k, v])

def lecture_donnees():
    """
    Renvoie le dictionnaire des résultats des tests

    EX : resultat_indiv={'gabriel'=[1,[],[]]} #Nom des individus : nb de tests, liste des temps de réac au buzzer, liste des temps au vibreur. Toute les 5 données on a un jour

    :return:
    """
    with open("D:/Kovelia/Documents/Cours/PREPA/A2/TIPE/CODE/DONNEES/BASE DONNEES/resultat_indiv.csv", mode="r") as fichier:
        lecteur = csv.reader(fichier)
        for ligne in lecteur:
            print(ligne)
        resultat_indiv= {ligne[0]: json.loads(ligne[1]) }
        return(resultat_indiv)
def analyse_indiv_rapid(prenom,donnees,type=''):
    """
        Calcule la moyenne, variance, écart-type des données
        return([[moy_buzzer,moy_vibreur,moy_tot],[var_buzzer,var_vibreur,var_tot],[ectype_buzzer,ectype_vibreur,ectype_tot]])

    :param prenom: 'tout' ou 'mael (par exemple)
    :param donnees: 'intens' ou quelconque (Intens pour le gradient d'intensité)
    :param type: 'moy','var','ectype'
    :return:
    """

    if donnees=='intens':
        if prenom.lower()=='tout':
            donnes_dico={}
            for nom in res_gradient_vib:
                if type.lower()=='moy' or type.lower()=='moyenne':
                    donnes_dico[nom]={key: round(np.mean(values),2) for key, values in res_gradient_vib[nom].items()}
                elif type.lower() == 'var' or type.lower() == 'variance':
                    donnes_dico[nom] = {key: round(np.var(values),2) for key, values in res_gradient_vib[nom].items()}
                elif type.lower() == 'ectype' or type.lower() == 'ecart type':
                    donnes_dico[nom] = {key: round(np.std(values),2) for key, values in res_gradient_vib[nom].items()}
        return(donnes_dico)
    else:
        if prenom.lower() == 'tout':
            res_global=[[],[]]
            for nom in resultat_indiv:
                res_global[0].extend(resultat_indiv[nom][1][:])
                res_global[1].extend(resultat_indiv[nom][2][:])
            if type.lower()=='moy' or type.lower()=='moyenne':
                moy_buzzer = np.mean(res_global[0])
                moy_vibreur = np.mean(res_global[1])
                moy_tot = np.mean(resultat_indiv[prenom][1] + res_global[1])
                return([moy_buzzer,moy_vibreur,moy_tot])
            elif type.lower()=='var' or type.lower()=='variance':
                var_buzzer = np.var(res_global[0])
                var_vibreur = np.var(res_global[1])
                var_tot = np.var(res_global[0] + res_global[1])
                return([var_buzzer,var_vibreur,var_tot])
            elif type.lower() == 'ectype' or type.lower() == 'ecart type':
                ectype_buzzer = np.std(res_global[0])
                ectype_vibreur = np.std(res_global[1])
                ectype_tot = np.std(res_global[0] + res_global[1])
                return([ectype_buzzer,ectype_vibreur,ectype_tot])
        else:
            if type.lower()=='moy' or type.lower()=='moyenne':
                moy_buzzer = np.mean(resultat_indiv[prenom][1])
                moy_vibreur = np.mean(resultat_indiv[prenom][1])
                moy_tot = np.mean(resultat_indiv[prenom][1] + resultat_indiv[prenom][1])
                return([moy_buzzer,moy_vibreur,moy_tot])
            elif type.lower()=='var' or type.lower()=='variance':
                var_buzzer = np.var(resultat_indiv[prenom][1])
                var_vibreur = np.var(resultat_indiv[prenom][1])
                var_tot = np.var(resultat_indiv[prenom][1] + resultat_indiv[prenom][1])
                return([var_buzzer,var_vibreur,var_tot])
            elif type.lower() == 'ectype' or type.lower() == 'ecart type':
                ectype_buzzer = np.std(resultat_indiv[prenom][1])
                ectype_vibreur = np.std(resultat_indiv[prenom][1])
                ectype_tot = np.std(resultat_indiv[prenom][1] + resultat_indiv[prenom][1])
                return([ectype_buzzer,ectype_vibreur,ectype_tot])
def graph_evojour(prenom='TOUT',type='TOUT'):
    """
    prenom = Nom de l'individu ou TOUT si pour tout les individus

    type = Buzzer, Vibreur, Tout

    Trace le graphique de la progression de chaque individu de façon individuelle par jour (Avec écart type)

    Et renvoie la moyenne de l'évolution par jour de prenom
    """

    liste_donnees = [[], []]  # indice 0 : Temps du buzzer ; Indice 1 : Temps du vibreur
    liste_moyennejour = [] #liste des moyennes par jour
    tdrevol_jour = [] #évolution des moyennes par jour
    liste_ectypejour=[]
    eccevol_jour=[]
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    nb_participant=6
    if prenom.upper() != 'TOUT':
        nb_jour = len(resultat_indiv[prenom.lower()][1])//5
        nb_participant=1
    else:
        nb_jour = len(resultat_indiv[list(resultat_indiv.keys())[0]][1])//5
    liste_jour = np.arange(nb_jour) + 1
    if prenom.upper() != 'TOUT':
        liste_donnees[0]+=resultat_indiv[prenom.lower()][1]
        liste_donnees[1]+=resultat_indiv[prenom.lower()][2]
    else:
        for i in range(nb_jour):
            for nom in resultat_indiv:
                liste_donnees[0]+=resultat_indiv[nom][1][5*i:5*i+5]
                liste_donnees[1]+=resultat_indiv[nom][2][5*i:5*i+5]
    if type.upper()=='TOUT' or type.upper()=='2' or type.upper()=='buzzer + vibreur' or type.upper()=='vibreur + buzzer':
        for i in range(nb_jour):
            moy_jouri = np.mean(liste_donnees[0][5*i*nb_participant:(5*i+5)*nb_participant] + liste_donnees[1][5*i*nb_participant:(5*i+5)*nb_participant])
            ec_jouri = np.std(liste_donnees[0][5*i*nb_participant:(5*i+5)*nb_participant] + liste_donnees[1][5*i*nb_participant:(5*i+5)*nb_participant])
            liste_moyennejour.append(moy_jouri) #commence au jour 0 (1er jour de test)
            tdrevol_jour.append(liste_moyennejour[i]-liste_moyennejour[i-1])
            liste_ectypejour.append(ec_jouri)
            eccevol_jour.append(liste_ectypejour[i]-np.mean(liste_ectypejour))
            axs[0].errorbar(i+1, liste_moyennejour[i], yerr=liste_ectypejour[i]/np.sqrt(5*nb_participant*2), xerr=0, linewidth=2,
                            capsize=6,c='black')
            axs[1].errorbar(i + 1, tdrevol_jour[i], yerr=liste_ectypejour[i] / np.sqrt(5 * nb_participant), xerr=0,
                            linewidth=2, capsize=6, c='black')
    elif type.upper()=='BUZZER' or type.upper()=='SON' or type.upper()=='BRUIT' or type.upper()=='BIP':
        for i in range(nb_jour):
            moy_jouri = np.mean(liste_donnees[0][5*i*nb_participant:(5*i+5)*nb_participant])
            ec_jouri = np.std(liste_donnees[0][5*i*nb_participant:(5*i+5)*nb_participant])
            liste_moyennejour.append(moy_jouri)  # commence au jour 0 (1er jour de test)
            tdrevol_jour.append(liste_moyennejour[i] - liste_moyennejour[i - 1])
            liste_ectypejour.append(ec_jouri)
            eccevol_jour.append(liste_ectypejour[i] - np.mean(liste_ectypejour))
            axs[0].errorbar(i+1, liste_moyennejour[i], yerr=liste_ectypejour[i] / np.sqrt(5 * nb_participant), xerr=0,fmt='o', linewidth=2, capsize=6,c='r')  # TODO : A calculer les incertitudes vibreur + buzzer
            axs[1].errorbar(i + 1, tdrevol_jour[i], yerr=liste_ectypejour[i] / np.sqrt(5 * nb_participant), xerr=0,
                            linewidth=2, capsize=6, c='black')
    elif type.upper()=='VIBREUR' or type.upper()=='MOTEUR' or type.upper()=='VIBRATION' or type.upper()=='MANETTE':
        for i in range(nb_jour):
            moy_jouri = np.mean(liste_donnees[1][5*i*nb_participant:(5*i+5)*nb_participant])
            ec_jouri = np.std(liste_donnees[1][5*i*nb_participant:(5*i+5)*nb_participant])
            analyse_indiv_rapid(prenom)
            liste_moyennejour.append(moy_jouri)  # commence au jour 0 (1er jour de test)
            tdrevol_jour.append(liste_moyennejour[i] - liste_moyennejour[i - 1])
            liste_ectypejour.append(ec_jouri)
            eccevol_jour.append(liste_ectypejour[i] - np.mean(liste_ectypejour))
            axs[0].errorbar(i+1, liste_moyennejour[i], yerr=liste_ectypejour[i] / np.sqrt(5 * nb_participant), xerr=0, linewidth=2,capsize=6,c='black')
            axs[1].errorbar(i + 1, tdrevol_jour[i], yerr=liste_ectypejour[i] / np.sqrt(5 * nb_participant), xerr=0, linewidth=2, capsize=6, c='black')
    if prenom.upper()=='TOUT':
        prenom="l'ensemble des participants"



    axs[0].grid()
    axs[0].set_ylim((0, 350))
    axs[0].set_xticks(np.arange(0, nb_jour + 2, 1))
    axs[0].set_xlabel("Jour d'étude",fontsize=17)
    axs[0].set_ylabel('Temps de réaction (ms)',fontsize=17)
    axs[0].plot(liste_jour, liste_moyennejour,marker='o', linestyle='-', c='red', label='Temps de réaction moyens de ' + prenom)
    axs[0].plot(liste_jour, liste_ectypejour, marker='o', linestyle='-', c='b',label="Écart type moyens de " + prenom)
    axs[0].legend(fontsize=15)
    axs[0].set_title("Évolution de la moyenne du temps de réaction par jour", fontsize=22)

    axs[1].grid()
    axs[1].set_ylim((-80, 80))
    axs[1].set_xticks(np.arange(0, nb_jour + 2, 1))
    axs[1].set_xlabel("Jour d'étude",fontsize=17)
    axs[1].set_ylabel('Delta du temps de réaction (ms)',fontsize=17)
    axs[1].plot(liste_jour, tdrevol_jour, marker='o',linestyle='-', c='orange', label='Delta du temps de réaction de ' + prenom)
    axs[1].plot(liste_jour, eccevol_jour, marker='o', linestyle='-', c='b',label="Delta de l'écart type de " + prenom)
    axs[1].legend(fontsize=15)
    axs[1].set_title("Évolution du delta de temps de réaction par jour", fontsize=22)

    moy_tdrevol_jour = np.mean(tdrevol_jour[1:])
    plt.tight_layout()
    plt.show()
    print("La moyenne de l'évolution du tdr par jour de "+prenom+" est de " +str(round(moy_tdrevol_jour,1))+' ms.')
    print("L'evolution du temps de réaction depuis le début est de " + str(round(liste_moyennejour[0]-liste_moyennejour[-1],1))+ ' ms.')
    print(liste_ectypejour[0])
    print(liste_ectypejour[0] - liste_ectypejour[-1])

def inverse_moyenne_tdr_intens(data):
    '''
    Créer un dictionnaire de l'inverse de la moyenne des résultats aux tests pour chaque rapport cyclique
    :param data: Dictionnaire des résultats en fonctions des rapports cycliques
    :return:
    '''
    return {key: 1 / np.mean(values) for key, values in data.items()}
def normaliser(x):
    '''
    Permet de normer les données pour ne pas avoir des valeurs trop élevées lors de la descente de gradient
    '''
    return (x - np.mean(x)) / np.std(x)

def cout(theta, x_biais, y):
    '''
    Fonction cout de la descente de gradient
    :param theta: Paramètres liant x_b et y (Ils évoluent durant la descente de gradient)
    :param x_b: Colonne des entrées avec une colonne de 1 pour prendre les biais
    :param y: Sortie
    :return:
    '''
    m = len(y)
    return (1 / (2 * m)) * np.sum((x_biais.dot(theta) - y.reshape(-1, 1)) ** 2)
def descente_gradient(rapport_cyclique, tdr, alpha=0.01, n_iterations=1000):
    '''
    Fonction effectuant la descente de gradient sur un test de données
    :param rapport_cyclique: Entrées
    :param tdr: Sortie
    :param alpha: Taux d'apprentissage
    :param n_iterations: Nombres d'itérations de la descente de gradient
    :return:
    '''
    rapport_cyclique_norm = normaliser(rapport_cyclique)
    tdr_norm = normaliser(tdr)
    nb_points = len(rapport_cyclique_norm)
    theta = np.random.randn(2, 1)
    rapport_cyclique_b = np.c_[np.ones((nb_points, 1)), rapport_cyclique_norm.reshape(-1, 1)]
    for iteration in range(n_iterations):
        gradients = (2 / nb_points) * rapport_cyclique_b.T.dot(rapport_cyclique_b.dot(theta) - tdr_norm.reshape(-1, 1))
        theta = theta - alpha * gradients
    return theta, rapport_cyclique_b, np.mean(tdr), np.std(tdr)

def afficher_regression(noms_selectionnes=None,comparaison=False):
    '''
    Permet d'afficher sur un graphique les regressions en 1/x du tests de données. Il peut aussi comparer sur un même graphique différentes régressions.
    Et peut afficher la régression de l'ensemble des données
    :param noms_selectionnes: Liste des noms qui doivent être pris en compte pour la/les régressions (Si on les compare ou on les regroupe)
    :param comparaison: False --> Pas de comparaison, donc si plusieurs nom, on a une régression sur l'ensemble des valeurs de noms_selectionnes. True --> Comparaison entre les régressions de noms_selectionnes.
    :return:
    '''
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    ec_intens=[]
    if noms_selectionnes is None:
        noms_selectionnes = list(res_gradient_vib.keys())

    combined_tdr = []
    combined_rapport_cyclique = []
    ec_i=[]
    points = np.linspace(1, 1500, 1000)
    for nom in noms_selectionnes:
        moy_tdr = inverse_moyenne_tdr_intens(res_gradient_vib[nom])
        rapport_cyclique = np.array(list(moy_tdr.keys()))
        tdr = np.array(list(moy_tdr.values()))
        for keys in list(res_gradient_vib[nom].keys()):
            moy_ec_intens[keys].extend(res_gradient_vib[nom][keys])
            ec_i=np.std(res_gradient_vib[nom][keys])
        # Tracer les points de données d'origine (TDR inversé)
        line, =axs[0].plot(rapport_cyclique, 1/tdr, ".", label=f"{nom.capitalize()} data")
        couleur_point=line.get_color()
        axs[0].errorbar(rapport_cyclique, 1/tdr, yerr = ec_i / np.sqrt(len(res_gradient_vib[nom][440])), xerr = 0, fmt = 'none', linewidth = 1, capsize = 3,c=couleur_point)
        # Accumuler les données pour la combinaison
        combined_tdr.extend(tdr)
        combined_rapport_cyclique.extend(rapport_cyclique)
        if comparaison:
            # Tracer la courbe de régression pour un seul participant
            theta, rapport_cyclique_b, mean_tdr, std_tdr = descente_gradient(rapport_cyclique, tdr)
            # Expression de la fonction tracée
            m_i = theta[1][0] * std_tdr / np.std(rapport_cyclique)  # Coefficient de pente
            b_i = (theta[0][0] * std_tdr + mean_tdr) - m_i * np.mean(rapport_cyclique)  # Ordonnée à l'origine
            axs[0].plot(points, 1 / (m_i * points + b_i), label=f"{nom.capitalize()}")
    for keys in moy_ec_intens.keys():
        ec_intens.append(np.std(moy_ec_intens[keys]))

    if len(noms_selectionnes) == 1:
        # Tracer la courbe de régression pour un seul participant
        theta, rapport_cyclique_b, mean_tdr, std_tdr = descente_gradient(rapport_cyclique, tdr)
        # Expression de la fonction tracée
        m = theta[1][0] * std_tdr / np.std(rapport_cyclique)  # Coefficient de pente
        b = (theta[0][0] * std_tdr + mean_tdr) - m * np.mean(rapport_cyclique)  # Ordonnée à l'origine
    elif len(noms_selectionnes) > 1:
        # Tracer la courbe de régression combinée pour plusieurs participants
        combined_tdr = np.array(combined_tdr)
        combined_rapport_cyclique = np.array(combined_rapport_cyclique)
        theta_combined, rapport_cyclique_b_combined, mean_combined_tdr, std_combined_tdr = descente_gradient(
            combined_rapport_cyclique, combined_tdr)
        # Expression de la fonction tracée
        m = theta_combined[1][0] * std_combined_tdr / np.std(combined_rapport_cyclique)  # Coefficient de pente
        b = (theta_combined[0][0] * std_combined_tdr + mean_combined_tdr) - m * np.mean(
            combined_rapport_cyclique)  # Ordonnée à l'origine

    axs[0].plot(points,1/(m*points+b),c='b',label='Régression linéaire par descente de gradient')
    axs[0].plot(points,100*points/points,c='r',label='Limite théorique du temps de réaction')
    axs[0].set_xlabel("Rapport cyclique (nb binaire 0-1023)",fontsize=17)
    axs[0].set_ylabel("Temps de réaction (ms)",fontsize=17)
    axs[0].set_ylim((100,701))
    axs[0].set_xlim(200,1300)
    axs[0].set_xticks(np.arange(200,1300,80))
    axs[0].set_yticks(np.arange(40,700,60))
    axs[0].legend(fontsize=14)
    axs[0].grid()
    axs[0].set_title(f"Temps de réaction en fonction du rapport cyclique",fontsize=22)

    axs[1].plot(rapport_cyclique, ec_intens, c='b',label='Écarts types moyens')
    axs[1].set_xlabel("Rapport cyclique (nb binaire 0-1023)",fontsize=17)
    axs[1].set_ylabel("Écarts types moyens (ms)",fontsize=17)
    axs[1].set_ylim((0, 120))
    axs[1].set_xlim(400, 1040)
    axs[1].set_xticks(np.arange(400, 1040, 40))
    axs[1].set_yticks(np.arange(0, 120, 10))
    axs[1].legend(fontsize=18)
    axs[1].grid()
    axs[1].set_title(f"Écarts types moyens en fonction du rapport cyclique",fontsize=22)
    plt.tight_layout()
    plt.show()
    print(ec_intens[0])
    print(ec_intens[0]-ec_intens[-1])
    print("L'équation de la courbe est " + f"y =1/({m:.2e}x + {b:.2e})")