//DesignModeler JScript, version: ANSYS DesignModeler 2019 R2 (Apr 16 2019, 11:11:40; 19,2019,106,1) SV4
//Created via: "Write Script: Sketch(es) of Active Plane"
// Written to: D:\summer_research\Real\assembly\concave.js
//         On: 07/10/19, 11:36:11
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
p.Sk2 = p.Plane.NewSketch();
p.Sk2.Name = "Sketch2";

//Edges
with (p.Sk2)
{
  p.Ln13 = Line(-0.05000000, 0.15000000, 0.05000000, 0.15000000);
  p.Ln14 = Line(0.05000000, 0.15000000, 0.05000000, -0.15000000);
  p.Ln15 = Line(0.05000000, -0.15000000, -0.05000000, -0.15000000);
  p.Ln16 = Line(-0.05000000, -0.15000000, -0.05000000, 0.15000000);
}

//Dimensions and/or constraints
with (p.Plane)
{
  //Dimensions
  var dim;
  dim = HorizontalDim(p.Ln16.End, -0.05000000, 0.15000000, 
    p.Ln14.Base, 0.05000000, 0.15000000, 
    0.03564103, 0.21576735);
  if(dim) dim.Name = "H1";
  dim = HorizontalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln16.End, -0.05000000, 0.15000000, 
    -0.03114286, 0.24436185);
  if(dim) dim.Name = "H2";
  dim = VerticalDim(p.Ln13.Base, -0.05000000, 0.15000000, 
    p.Ln15.End, -0.05000000, -0.15000000, 
    -0.31132264, -0.12848517);
  if(dim) dim.Name = "V3";
  dim = VerticalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln15.End, -0.05000000, -0.15000000, 
    -0.19551491, -0.11561265);
  if(dim) dim.Name = "V4";

  //Constraints
  HorizontalCon(p.Ln13);
  HorizontalCon(p.Ln15);
  VerticalCon(p.Ln14);
  VerticalCon(p.Ln16);
  CoincidentCon(p.Ln13.End, 0.05000000, 0.15000000, 
                p.Ln14.Base, 0.05000000, 0.15000000);
  CoincidentCon(p.Ln14.End, 0.05000000, -0.15000000, 
                p.Ln15.Base, 0.05000000, -0.15000000);
  CoincidentCon(p.Ln15.End, -0.05000000, -0.15000000, 
                p.Ln16.Base, -0.05000000, -0.15000000);
  CoincidentCon(p.Ln16.End, -0.05000000, 0.15000000, 
                p.Ln13.Base, -0.05000000, 0.15000000);
}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps1 = planeSketchesOnly (new Object());

//Finish
agb.Regen(); //To insure model validity
//End DM JScript
