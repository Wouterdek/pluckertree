# Plückertree - Fast nearest neighbouring line lookup

This is the code written for my master thesis with the dutch title "Plückerbomen: efficiënt zoeken van k dichtstbijzijnde rechten rond punt in 3D, met toepassing in photonmapping". (in english: "Plückertrees: efficient search of k nearest lines around point in 3D, with application in photonmapping")

Main takeaway of this paper: using a tree based on plückercoordinates, it is possible to find the lines with the shortest distance to an arbitrary 3D point in about O(n^0.608) time while taking up only O(n) space. The paper shows comparable performance for similar problems such as rays instead of lines, or using distance to line-plane intersection instead of line-point.

This is research code, so definitely not production ready.
You can find the PDF of my thesis [here](Thesis.pdf).