//DesignModeler JScript, version: ANSYS DesignModeler 2019 R2 (Apr 16 2019, 11:11:40; 19,2019,106,1) SV4
//Created via: "Write Script: Sketch(es) of Active Plane"
// Written to: D:\summer_research\T\create_T.js
//         On: 08/12/19, 18:50:08
//Using:
//  agb ... pointer to batch interface


//Note:
// You may be able to re-use below JScript function via cut-and-paste;
// however, you may have to re-name the function identifier.
//

function planeSketchesOnly (p)
{

//Plane
p.Plane  = agb.GetYZPlane();
p.Origin = p.Plane.GetOrigin();
p.XAxis  = p.Plane.GetYAxis();
p.YAxis  = p.Plane.GetXAxis();

//Sketch
p.Sk1 = p.Plane.NewSketch();
p.Sk1.Name = "Sketch1";

//Edges
with (p.Sk1)
{
  p.Ln8 = Line(0.15, -0.2, 0.15, 0);
  p.Ln9 = Line(0.15, 0, 0.45, 0);
  p.Ln10 = Line(0.45, 0, 0.45, 0.2);
  p.Ln11 = Line(0.45, 0.2, -0.45, 0.2);
  p.Ln17 = Line(-0.45, 0.2, -0.45, 0);
  p.Ln18 = Line(-0.45, 0, -0.15, 0);
  p.Ln19 = Line(-0.15, 0, -0.15, -0.2);
  p.Ln20 = Line(-0.15, -0.2, 0.15, -0.2);
}

//Dimensions and/or constraints
with (p.Plane)
{
  //Constraints
  HorizontalCon(p.Ln9);
  HorizontalCon(p.Ln11);
  HorizontalCon(p.Ln18);
  HorizontalCon(p.Ln20);
  VerticalCon(p.Ln8);
  VerticalCon(p.Ln17);
  CoincidentCon(p.Ln8.End, 0.15, 0, 
                p.Ln9.Base, 0.15, 0);
  CoincidentCon(p.Ln9.End, 0.45, 0, 
                p.Ln10.Base, 0.45, 0);
  CoincidentCon(p.Ln10.End, 0.45, 0.2, 
                p.Ln11.Base, 0.45, 0.2);
  CoincidentCon(p.Ln17.Base, -0.45, 0.2, 
                p.Ln11.End, -0.45, 0.2);
  CoincidentCon(p.Ln18.Base, -0.45, 0, 
                p.Ln17.End, -0.45, 0);
  CoincidentCon(p.Ln19.Base, -0.15, 0, 
                p.Ln18.End, -0.15, 0);
  CoincidentCon(p.Ln20.Base, -0.15, -0.2, 
                p.Ln19.End, -0.15, -0.2);
  CoincidentCon(p.Ln20.End, 0.15, -0.2, 
                p.Ln8.Base, 0.15, -0.2);
}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps1 = planeSketchesOnly (new Object());

//Finish
agb.Regen(); //To insure model validity


var ext1 = agb.Extrude(agc.Add, ps1.Sk1, agc.DirNormal, agc.ExtentFixed, 0.2,
  agc.ExtentFixed, 0.0, agc.No, 0.0, 0.0);

agb.Regen();


//End DM JScript
