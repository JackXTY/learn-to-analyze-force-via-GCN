//DesignModeler JScript, version: ANSYS DesignModeler 2019 R2 (Apr 16 2019, 11:11:40; 19,2019,106,1) SV4
//Created via: "Write Script: Sketch(es) of Active Plane"
// Written to: D:\summer_research\Real\assembly\creat_surface_for_cutting.js
//         On: 07/10/19, 12:21:53
//Using:
//  agb ... pointer to batch interface


//Note:
// You may be able to re-use below JScript function via cut-and-paste;
// however, you may have to re-name the function identifier.
//

function planeSketchesOnly (p)
{

//Plane
p.Plane  = agb.GetActivePlane();
p.Origin = p.Plane.GetOrigin();
p.XAxis  = p.Plane.GetXAxis();
p.YAxis  = p.Plane.GetYAxis();

//Sketch
p.Sk1 = p.Plane.NewSketch();
p.Sk1.Name = "Sketch1";

//Edges
with (p.Sk1)
{
  p.Ln7 = Line(-0.15000000, 0.05000000, 0.15000000, 0.05000000);
  p.Ln8 = Line(0.15000000, 0.05000000, 0.15000000, -0.05000000);
  p.Ln9 = Line(0.15000000, -0.05000000, -0.15000000, -0.05000000);
  p.Ln10 = Line(-0.15000000, -0.05000000, -0.15000000, 0.05000000);
}


//Sketch
p.Sk2 = p.Plane.NewSketch();
p.Sk2.Name = "Sketch2";

//Edges
with (p.Sk2)
{
  p.Ln11 = Line(-0.05000000, 0.15000000, 0.05000000, 0.15000000);
  p.Ln12 = Line(0.05000000, 0.15000000, 0.05000000, -0.15000000);
  p.Ln13 = Line(0.05000000, -0.15000000, -0.05000000, -0.15000000);
  p.Ln14 = Line(-0.05000000, -0.15000000, -0.05000000, 0.15000000);
}

//Dimensions and/or constraints
with (p.Plane)
{
  //Dimensions
  var dim;
  dim = HorizontalDim(p.Ln10.End, -0.15000000, 0.05000000, 
    p.Ln8.Base, 0.15000000, 0.05000000, 
    0.07269939, 0.20956459);
  if(dim) dim.Name = "H1";
  dim = HorizontalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln10.End, -0.15000000, 0.05000000, 
    -0.09527027, 0.12857347);
  if(dim) dim.Name = "H2";
  dim = HorizontalDim(p.Ln14.End, -0.05000000, 0.15000000, 
    p.Ln12.Base, 0.05000000, 0.15000000, 
    0.03564103, 0.21576735);
  if(dim) dim.Name = "H3";
  dim = HorizontalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln14.End, -0.05000000, 0.15000000, 
    -0.03114286, 0.24436185);
  if(dim) dim.Name = "H4";
  dim = VerticalDim(p.Ln9.End, -0.15000000, -0.05000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.35055857, 0.00789474);
  if(dim) dim.Name = "V3";
  dim = VerticalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.21154397, 0.02758241);
  if(dim) dim.Name = "V4";
  dim = VerticalDim(p.Ln11.Base, -0.05000000, 0.15000000, 
    p.Ln13.End, -0.05000000, -0.15000000, 
    -0.31132264, -0.12848517);
  if(dim) dim.Name = "V5";
  dim = VerticalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln13.End, -0.05000000, -0.15000000, 
    -0.19551491, -0.11561265);
  if(dim) dim.Name = "V6";

  //Constraints
  HorizontalCon(p.Ln7);
  HorizontalCon(p.Ln9);
  HorizontalCon(p.Ln11);
  HorizontalCon(p.Ln13);
  VerticalCon(p.Ln8);
  VerticalCon(p.Ln10);
  VerticalCon(p.Ln12);
  VerticalCon(p.Ln14);
  CoincidentCon(p.Ln7.End, 0.15000000, 0.05000000, 
                p.Ln8.Base, 0.15000000, 0.05000000);
  CoincidentCon(p.Ln8.End, 0.15000000, -0.05000000, 
                p.Ln9.Base, 0.15000000, -0.05000000);
  CoincidentCon(p.Ln9.End, -0.15000000, -0.05000000, 
                p.Ln10.Base, -0.15000000, -0.05000000);
  CoincidentCon(p.Ln10.End, -0.15000000, 0.05000000, 
                p.Ln7.Base, -0.15000000, 0.05000000);
  CoincidentCon(p.Ln11.End, 0.05000000, 0.15000000, 
                p.Ln12.Base, 0.05000000, 0.15000000);
  CoincidentCon(p.Ln12.End, 0.05000000, -0.15000000, 
                p.Ln13.Base, 0.05000000, -0.15000000);
  CoincidentCon(p.Ln13.End, -0.05000000, -0.15000000, 
                p.Ln14.Base, -0.05000000, -0.15000000);
  CoincidentCon(p.Ln14.End, -0.05000000, 0.15000000, 
                p.Ln11.Base, -0.05000000, 0.15000000);
}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps1 = planeSketchesOnly (new Object());

//Finish
agb.Regen(); //To insure model validity
//End DM JScript
