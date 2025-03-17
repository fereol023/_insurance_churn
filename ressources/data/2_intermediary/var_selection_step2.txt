
VALEURS MANQUANTES
--------------------------------------------

*- Par la suite on va regarder en detail les autres variables qui ont des valeurs manquantes :*
- `AGE` : 160k assures dont l'age est Nan - on peut essayer une imputation/(recalcul car on a la `DATE OF BIRTH` pas de null) dans la mesure ou ca ne represente que 5% du dataset. Mais attention parce qu'il y a des ages abberants (exple 112 ans pour un conducteur de voiture). On fera l'imputation seulement apres la dataviz.
- `INCOME`: 160k (meme nombre que age) - a voir si ce sont les memes - strategie d'imputation aussi sinon
- `HAS_CHILDREN`: 160k aussi sachant que les modalites yes/no sont bien renseignees - ne pas faire d'imputation
- `LENGTH_OF_RESIDENCE` et `HOME_OWNER` et `COLLEGE_DEGREE` et `GOOD_CREDIT`: 160k 
- `MARITAL STATUS`: une proportion de None differente (599k) - encoder par 'non renseigne'
- `HOME MARKET VALUE`: une proportion differente (357k) - tenter imputation si dataviz ok (car on regrade dans un meme perimetre geographique) - ou predire avec les caracteriqtiques geographiques

Conclusion partielle :
--------------------------------------------
*D'apres les colonnes qui manquent, on peut conclure que <u>les 160k sont des assures dont on n'a pas les vraiables socio demographiques</u>.<br>
On aurait pu tenter de predire la variable `HOME_MARKET_VALUE` a partir des autres mais il manque aussi les cacteriqties de la maison (nas).<br>
On peut tenter de faire une imputation pour garder la data coute que coute mais on ne fera pas. Plutot faire un drop car cela represente juste 5% du dataset.<br>
Une imputation ne rajouterait pas de la qualite au dataset.*
