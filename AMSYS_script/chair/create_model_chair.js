var Yes = agc.Yes;
var No  = agc.No;



//creat the first sketch
function planeSketchesYZOnly (p)
{

//Plane
p.Plane  = agb.GetYZPlane();
p.Origin = p.Plane.GetOrigin();
p.XAxis  = p.Plane.GetXAxis();
p.YAxis  = p.Plane.GetYAxis();

//Sketch
p.Sk1 = p.Plane.NewSketch();
p.Sk1.Name = "Sketch1";

//Edges
with (p.Sk1)
{
  p.Ln17 = Line(-0.1,  -0.1 , -0.03  , -0.1 );
  p.Ln18 = Line(-0.03, -0.1 , -0.03  , -0.04);
  p.Ln19 = Line(-0.03, -0.04, 0.13, -0.04);
  p.Ln20 = Line(0.13 ,  -0.04, 0.13, -0.1);

  p.Ln21 = Line(0.13 , -0.1, 0.2, -0.1);
  p.Ln22 = Line(0.2 , -0.1, 0.2, 0  );
  p.Ln23 = Line(0.2 , 0  , -0.1, 0  );
  p.Ln24 = Line(-0.1, 0  , -0.1, -0.1);
}

//Dimensions and/or constraints
with (p.Plane)
{
	/*
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
  dim = VerticalDim(p.Ln9.End, -0.15000000, -0.05000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.35055857, 0.00789474);
  if(dim) dim.Name = "V3";
  dim = VerticalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.21154397, 0.02758241);
  if(dim) dim.Name = "V4";
  */

  //Constraints
  HorizontalCon(p.Ln17);
  HorizontalCon(p.Ln19);
  HorizontalCon(p.Ln21);
  HorizontalCon(p.Ln23);
  VerticalCon(p.Ln18);
  VerticalCon(p.Ln20);
  VerticalCon(p.Ln22);
  VerticalCon(p.Ln24);
  CoincidentCon(p.Ln17.End,  -0.03,  -0.1, 
                p.Ln18.Base, -0.03,  -0.1);
  CoincidentCon(p.Ln18.End,  -0.03,  -0.04, 
                p.Ln19.Base, -0.03,  -0.04);
  CoincidentCon(p.Ln19.End,  0.13,-0.04, 
                p.Ln20.Base, 0.13,-0.04);
  CoincidentCon(p.Ln20.End,  0.13,-0.1, 
                p.Ln21.Base, 0.13,-0.1);
  CoincidentCon(p.Ln21.End,  0.2, -0.1, 
				p.Ln22.Base, 0.2, -0.1);
  CoincidentCon(p.Ln22.End,  0.2, 0, 
				p.Ln23.Base, 0.2, 0);
  CoincidentCon(p.Ln23.End, -0.1, 0, 
				p.Ln24.Base,-0.1, 0);
  CoincidentCon(p.Ln24.End, -0.1, -0.1, 
				p.Ln17.Base,-0.1, -0.1);

}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps1 = planeSketchesYZOnly (new Object());

//Finish
agb.Regen(); //To insure model validity

var ext1 = agb.Extrude(agc.Add, ps1.Sk1, agc.DirNormal, agc.ExtentFixed, 0.1125,
    agc.ExtentFixed, 0.0, agc.No, 0.0, 0.0);

agb.Regen();

