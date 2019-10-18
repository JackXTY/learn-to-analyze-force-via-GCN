//DesignModeler JScript, version: ANSYS DesignModeler 2019 R2 (Apr 16 2019, 11:11:40; 19,2019,106,1) SV4
//Created via: "Write Script: Sketch(es) of Active Plane"
// Written to: D:\summer_research\Unnamed.js
//         On: 06/16/19, 18:33:42
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
  p.Ln7 = Line(0.00000000, 0.00000000, 0.10000000, 0.00000000);
  p.Ln8 = Line(0.10000000, 0.00000000, 0.10000000, 0.30000000);
  p.Ln9 = Line(0.10000000, 0.30000000, 0.00000000, 0.30000000);
  p.Ln10 = Line(0.00000000, 0.30000000, 0.00000000, 0.00000000);
}

//Dimensions and/or constraints
with (p.Plane)
{
  //Constraints
  HorizontalCon(p.Ln7);
  HorizontalCon(p.Ln9);
  VerticalCon(p.Ln8);
  VerticalCon(p.Ln10);
  CoincidentCon(p.Ln7.End, 0.10000000, 0.00000000, 
                p.Ln8.Base, 0.10000000, 0.00000000);
  CoincidentCon(p.Ln8.End, 0.10000000, 0.30000000, 
                p.Ln9.Base, 0.10000000, 0.30000000);
  CoincidentCon(p.Ln9.End, 0.00000000, 0.30000000, 
                p.Ln10.Base, 0.00000000, 0.30000000);
  CoincidentCon(p.Ln10.End, 0.00000000, 0.00000000, 
                p.Ln7.Base, 0.00000000, 0.00000000);
  CoincidentCon(p.Ln7.Base, 0.00000000, 0.00000000, 
                p.Origin, 0.00000000, 0.00000000);
}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps1 = planeSketchesOnly (new Object());


//Extrude
var ext1 = agb.Extrude(agc.Add, ps1.Sk1, agc.DirNormal, agc.ExtentFixed, 0.06,
    agc.ExtentFixed, 0.0, agc.No, 0.0, 0.0);
ext1.SetDirection(0,0,0.1);


//Finish
agb.Regen(); //To insure model validity
//End DM JScript
