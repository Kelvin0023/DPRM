# Objects

Our set of objects is below.
- Sphere
- Cylinder
- Icosahedron
- Dodecahedron
- Cube

For each we have 2 versions of the mesh files (`_scaled.stl` and `_unscaled.stl`) to support different uses cases. When the user choses an object without providing the object size, a hand tuned version of the mesh is loaded. Else, an unscaled mesh where the farthest mesh is at a radius of 0.5 is used with provided object size as scale.

Note about hand tuned versions (`_scaled.stl`) of the mesh:

The sphere, cylinder (height=diameter), dodecahedron and icosahedron are all 75mm in diameter and cube is of length 60mm.

Mesh file source:

- Sphere: https://en.wikipedia.org/wiki/File:Sphere.stl
- Cylinder: Created with OpenSCAD using the following code
    ```
    $fn = 100;
    cylinder(h=1, r1=0.5, r2=0.5, center=true);
    ```
- Icosahedron: https://en.wikipedia.org/wiki/File:Icosahedron.stl
- Dodecahedron: https://en.wikipedia.org/wiki/File:Dodecahedron.stl
