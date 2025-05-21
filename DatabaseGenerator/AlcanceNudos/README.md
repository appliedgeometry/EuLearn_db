# This is a python script that calculates the reach for the tubular neighborhood of thickened knots

The output is one or two PLYs, the first corresponds to the 1D curve and the second to the triangulation of the resulting knot surface

If the curve has self-intersections or the reach exceeds the threshold, the second PLY is not generated

Input: 
- Number of knots
- Number of Lissajous knots cosine parameters
- Triplet for the frequencies in $(x,y,z)$
- Number of iterations
- Number of points to form the curve

The triplets are read from `AlcanceInicial/Ternas/Ternas.txt`

The script `AlcanceInicial/Ternas/GeneraTernas.py` generates the list of triplets `Ternas.txt` by selecting the range for $(n_x, n_y, n_z)$ to be relatively prime
