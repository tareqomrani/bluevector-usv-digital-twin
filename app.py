"""BLUEVECTOR | Autonomous USV Simulation & Digital Twin.

Run: streamlit run app.py
Self-contained app.py: Python UI embeds portable dynamics, powertrain,
linear algebra, and browser controllers plus hull parameters
for selectable 3-DOF / illustrative 6-DOF and Plotly/Three.js visualization.
No separate local engine files, model_parameters.json, or pages directory.

Engineering disclaimer: Demonstration-scale, unvalidated vessel parameters.
Not for real-vessel command, navigation, or COLREG compliance.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import welch
import streamlit as st
import streamlit.components.v1 as components


# -----------------------------------------------------------------------------
# Framework-neutral domain data and physics engines. Kept as named, independently
# extractable resources so another host can reuse them without Streamlit.
# The 3D renderers still require browser access to the Plotly/Three.js CDNs.
# -----------------------------------------------------------------------------
HULL = json.loads(r'''
{
  "length": 4.0,
  "beam": 1.6,
  "mass": 110.0,
  "Iz": 190.0,
  "m11": 150.0,
  "m22": 275.0,
  "m23": 38.0,
  "m33": 245.0,
  "Xu": 16.0,
  "Xuu": 24.0,
  "Yv": 125.0,
  "Yvv": 165.0,
  "Nr": 100.0,
  "Nrr": 155.0,
  "Dvr": 12.0,
  "arm": 0.55,
  "maxThrust": 160.0,
  "batteryWh": 1600.0,
  "maxMotorPower": 850.0,
  "hotelPower": 25.0,
  "motorTau": 0.45,
  "batteryResistance": 0.07,
  "motorThermalCapacity": 3200.0,
  "motorCooling": 2.8,
  "sixDof": {
    "heaveMass": 210.0,
    "rollInertia": 35.0,
    "pitchInertia": 210.0,
    "waterplaneArea": 2.1,
    "GMt": 0.32,
    "GMl": 0.75,
    "thrusterZ": -0.26,
    "heaveDamping": 245.0,
    "rollDamping": 24.0,
    "pitchDamping": 110.0,
    "heaveQuadratic": 390.0,
    "rollQuadratic": 25.0,
    "pitchQuadratic": 120.0,
    "hydroGeometry": {
      "mode": "twin-pontoon-trapezoid",
      "waterlineLength": 3.3,
      "pontoonWidth": 0.14,
      "pontoonCenterOffset": 0.49,
      "blockCoefficient": 0.78,
      "cgHeightAboveWaterline": 0.0,
      "pontoonFreeboard": 0.19
    },
    "rigidInertia": {
      "Ixx": 20.0,
      "Iyy": 145.0,
      "Izz": 190.0
    },
    "waveExcitationGain": 0.5
  },
  "powertrain": {
    "waterDensity": 1025.0,
    "propellerDiameter": 0.28,
    "shaftEfficiency": 0.78,
    "inducedFactor": 1.15,
    "motorIdlePower": 0.0,
    "minBusVoltage": 18.0,
    "cutoffSoc": 0.02
  }
}
''')
SIXDOF_ENGINE = r'''
/* BLUEVECTOR V9: finite-attitude six-axis reference simulator.
 * ENU/world z-up; body x-forward, y-port, z-up. Quaternion body -> ENU.
 * State nu = BODY VELOCITY RELATIVE TO WATER, omega = body angular rates.
 * Ground velocity = R(q) nu_linear + current_ENU.
 * M_RB nu_abs_dot + C_RB(nu_abs) nu_abs + M_A nu_rel_dot +
 * C_A(nu_rel) nu_rel + D(nu_rel) = tau_act + tau_hydro + tau_wave.
 * Rigid-body and added-mass terms are independently represented.
 * Hydrostatics integrate submerged volume and centroid for an IDEALIZED
 * 2-pontoon prism; the resulting gravity/buoyancy wrench is conservative
 * in still water, unlike uncoupled Euler-angle spring moments.
 * Wave forcing is EXOGENOUS linearized excitation, not CFD/validated RAO.
 */
(function(root,factory){const API=factory();if(typeof module!=='undefined'&&module.exports)module.exports=API;else root.Bluevector6DOF=API;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const G=9.80665,EPS=1e-12,TAU=2*Math.PI;
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>Math.sqrt(dot(a,a));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function multiplyQuat(a,b){return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];}
function unitQuat(q){const n=norm(q);return n>EPS?q.map(x=>x/n):[1,0,0,0];}
function fromEuler(roll,pitch,yaw){const a=Math.cos(roll/2),b=Math.sin(roll/2),c=Math.cos(pitch/2),d=Math.sin(pitch/2),e=Math.cos(yaw/2),f=Math.sin(yaw/2);return unitQuat([a*c*e+b*d*f,b*c*e-a*d*f,a*d*e+b*c*f,a*c*f-b*d*e]);}
function toEuler(q){const [w,x,y,z]=unitQuat(q);return {roll:Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),pitch:Math.asin(clamp(2*(w*y-z*x),-1,1)),yaw:wrap(Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))};}
function bodyToWorld(q,v){const Q=unitQuat(q),a=Q.slice(1),t=cross(a,v).map(x=>2*x),c=cross(a,t);return v.map((x,i)=>x+Q[0]*t[i]+c[i]);}
function worldToBody(q,v){const Q=unitQuat(q);return bodyToWorld([Q[0],-Q[1],-Q[2],-Q[3]],v);}
function advanceQuaternion(q,omega,dt){const theta=norm(omega)*dt,factor=theta<1e-8?dt/2:Math.sin(theta/2)/(theta/dt);return unitQuat(multiplyQuat(q,[Math.cos(theta/2),...omega.map(w=>w*factor)]));}
// Bearing is direction TO, clockwise from north, all angles in radians internally.
function waveAt(x,y,t,wave={}){const H=Math.max(0,Number(wave.height)||0),T=Math.max(.8,Number(wave.period)||3.5),bearing=(Number(wave.bearing)||0)*Math.PI/180,east=Math.sin(bearing),north=Math.cos(bearing),omega=TAU/T,k=omega*omega/G,phase=k*(east*x+north*y)-omega*t,A=H/2;return {z:A*Math.sin(phase),slopeE:A*k*Math.cos(phase)*east,slopeN:A*k*Math.cos(phase)*north,omega,k,phase,steepness:A*k};}
function hydrostatics(hull){
 const e=hull.sixDof||{},geo=e.hydroGeometry||{};
 const L=Number(geo.waterlineLength),b=Number(geo.pontoonWidth),off=Number(geo.pontoonCenterOffset),Cb=Number(geo.blockCoefficient),CGz=Number(geo.cgHeightAboveWaterline),freeboard=Number(geo.pontoonFreeboard??.19);
 if(!(L>0&&L<=hull.length&&b>0&&Cb>.5&&Cb<=1&&off>=0&&off+b/2<=hull.beam/2&&Number.isFinite(CGz)&&freeboard>0))throw Error('invalid pontoon geometry');
 const rho=Number(geo.waterDensity)||1025,displacement=hull.mass/rho,waterplane=2*L*b,draft=displacement/(waterplane*Cb),bottomWidthRatio=2*Cb-1;
 const Itrans=2*(L*b**3/12+L*b*off**2),Ilong=2*(b*L**3/12);
 const cbDepth=draft*(4*Cb-1)/(6*Cb),BG=CGz+cbDepth;
 const GMt=Itrans/displacement-BG,GMl=Ilong/displacement-BG;
 if(!(GMt>0&&GMl>0&&draft>0))throw Error('nonpositive restoring hydrostatics');
 return {rho,displacement_m3:displacement,draft_m:draft,waterplaneArea_m2:waterplane,
 transverseWaterplaneInertia_m4:Itrans,longitudinalWaterplaneInertia_m4:Ilong,centerBuoyancyDepth_m:cbDepth,
 bottomWidthRatio,GMt_m:GMt,GMl_m:GMl,CGz_m:CGz,BG_m:BG,freeboard_m:freeboard,
 Kz:rho*G*waterplane,Kroll:hull.mass*G*GMt,Kpitch:hull.mass*G*GMl,
 source:'FINITE_ATTITUDE_TRAPEZOIDAL_PONTOON_PROXY_UNCALIBRATED'};
}
function modelParams(hull){
 const e=hull.sixDof||{},rigid=e.rigidInertia||{};
 const M=[hull.m11,hull.m22,e.heaveMass||210,e.rollInertia||35,e.pitchInertia||210,hull.m33];
 const MRB=[hull.mass,hull.mass,hull.mass,rigid.Ixx,rigid.Iyy,rigid.Izz];
 const MA=M.map((x,i)=>x-MRB[i]);
 const D=[hull.Xu,hull.Yv,e.heaveDamping??245,e.rollDamping??24,e.pitchDamping??110,hull.Nr];
 const Q=[hull.Xuu,hull.Yvv,e.heaveQuadratic??390,e.rollQuadratic??25,e.pitchQuadratic??120,hull.Nrr];
 const crossSwayYaw=hull.m23||0,crossDamping=hull.Dvr||0;
 if(M.some(x=>!Number.isFinite(x)||x<=0)||MRB.some(x=>!Number.isFinite(x)||x<=0)||MA.some(x=>!Number.isFinite(x)||x<0)||MA[1]*MA[5]<=crossSwayYaw**2)throw Error('invalid mass/added-mass tensors');
 if(D.some(x=>!Number.isFinite(x)||x<0)||D[1]*D[5]<crossDamping**2)throw Error('non-positive dissipation');
 const h=hydrostatics(hull);
 return {M,MRB,MA,D,Q,crossSwayYaw,crossDamping,hydro:h,
 Kz:h.Kz,Kroll:h.Kroll,Kpitch:h.Kpitch,thrusterZ:e.thrusterZ??-.26,
 waveExcitationGain:e.waveExcitationGain??.5};
}
// Gauss integration by symmetric segments, split at trapezoid's keel-width edge;
// exact in the calm upright configuration and smooth for moderate heel/trim.
const GL4_X=[-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526];
const GL4_W=[.3478548451374539,.6521451548625461,.6521451548625461,.3478548451374539];
const columnCache=new WeakMap();
function hydroColumns(hull,par){
 const cache=columnCache.get(hull);
 const key=[hull.sixDof.hydroGeometry.waterlineLength,hull.sixDof.hydroGeometry.pontoonWidth,hull.sixDof.hydroGeometry.pontoonCenterOffset,par.hydro.draft_m,par.hydro.bottomWidthRatio].join('|');
 if(cache&&cache.key===key)return cache.columns;
 const g=hull.sixDof.hydroGeometry,h=par.hydro,L=g.waterlineLength,b=g.pontoonWidth,r=h.bottomWidthRatio,offset=g.pontoonCenterOffset,cols=[];
 const Nx=24,dx=L/Nx;
 // Quadrature over y positive half-section, then mirror and add both pontoons.
 for(let side of [-1,1])for(let i=0;i<Nx;i++){
  const x=-L/2+dx*(i+.5);
  for(const [lo,hi] of [[0,r*b/2],[r*b/2,b/2]])for(let j=0;j<4;j++){
   const ya=(lo+hi)/2+(hi-lo)/2*GL4_X[j],weight=dx*(hi-lo)/2*GL4_W[j];
   const bottom=ya<=r*b/2+1e-15?-h.draft_m:-h.draft_m+h.draft_m*(2*ya/b-r)/(1-r);
   for(let sign of [-1,1])cols.push([x,side*offset+sign*ya,bottom,weight]);
  }
 }
 columnCache.set(hull,{key,columns:cols});
 return cols;
}
// Forces from gravity + submerged-volume buoyancy, both in BODY axes.
// Displaced-volume moments are calculated with the same quadrature as energy.
function finiteHydrostaticWrench(state,hull,par){
 const Q=state.attitude||fromEuler(state.roll||0,state.pitch||0,state.psi||0),h=par.hydro;
 const zRows=[bodyToWorld(Q,[1,0,0])[2],bodyToWorld(Q,[0,1,0])[2],bodyToWorld(Q,[0,0,1])[2]];
 if(zRows[2]<.12)return {valid:false,reason:'hull upside down/outside upright proxy',tau:[0,0,0,0,0,0],potential_J:NaN,displacedVolume_m3:0};
 let V=0,mx=0,my=0,mz=0,immersedMomentWorldZ=0;
 for(const [x,y,zBase,area] of hydroColumns(hull,par)){
  const zLimit=(-state.z-zRows[0]*x-zRows[1]*y)/zRows[2];
  const zTop=Math.min(h.freeboard_m,zLimit);
  if(zTop<=zBase)continue;
  const dv=area*(zTop-zBase),zm=area*(zTop*zTop-zBase*zBase)/2;
  V+=dv;mx+=x*dv;my+=y*dv;mz+=zm;
  immersedMomentWorldZ+=state.z*dv+zRows[0]*x*dv+zRows[1]*y*dv+zRows[2]*zm;
 }
 const B=V>EPS?[mx/V,my/V,mz/V]:[0,0,0],Gbody=[0,0,h.CGz_m];
 const Fb=worldToBody(Q,[0,0,h.rho*G*V]),Fg=worldToBody(Q,[0,0,-hull.mass*G]);
 const MB=cross(B,Fb),MG=cross(Gbody,Fg);
 const tau=[Fb[0]+Fg[0],Fb[1]+Fg[1],Fb[2]+Fg[2],...MB.map((x,i)=>x+MG[i])];
 const potential_J=hull.mass*G*(state.z+zRows[2]*h.CGz_m)-h.rho*G*immersedMomentWorldZ;
 const waterlineValid=V>EPS&&V<2*hull.sixDof.hydroGeometry.waterlineLength*hull.sixDof.hydroGeometry.pontoonWidth*(h.draft_m+h.freeboard_m);
 return {valid:waterlineValid,tau,potential_J,displacedVolume_m3:V,centerBuoyancyBody_m:B,
 buoyantForce_N:h.rho*G*V,draft_m:h.draft_m,submergenceLimit:!waterlineValid};
}
function initialize(boat){boat.z=boat.z||0;boat.w=boat.w||0;boat.p=boat.p||0;boat.q=boat.q||0;boat.roll=boat.roll||0;boat.pitch=boat.pitch||0;boat.attitude=fromEuler(boat.roll,boat.pitch,boat.psi);boat.waveHeight=0;boat.waveElevation=0;boat.rollEquilibrium=0;boat.pitchEquilibrium=0;return boat;}
function matVec(diagonal,crossSwayYaw,nu){const m=diagonal.map((x,i)=>x*nu[i]);m[1]+=crossSwayYaw*nu[5];m[5]+=crossSwayYaw*nu[1];return m;}
function coriolisAction(nu,momentum){const v=nu.slice(0,3),omega=nu.slice(3),p=momentum.slice(0,3),h=momentum.slice(3),f=cross(omega,p),m=cross(v,p).map((x,i)=>x+cross(omega,h)[i]);return [...f,...m];}
function currentAtState(state,inputs){const c=inputs.current||{};return [Number(c.x)||0,Number(c.y)||0,0];}
function evaluateDynamics(state,inputs,par,hull,t){
 const Q=state.attitude,rel=[state.u,state.v,state.w||0,state.p||0,state.q||0,state.r],omega=rel.slice(3);
 const currentWorld=currentAtState(state,inputs),currentBody=worldToBody(Q,currentWorld);
 const absolute=[rel[0]+currentBody[0],rel[1]+currentBody[1],rel[2]+currentBody[2],...omega];
 const ground=bodyToWorld(Q,rel.slice(0,3)).map((x,i)=>x+currentWorld[i]);
 const grad=inputs.currentDerivatives||{},
  dcE=(grad.dtE||0)+(grad.dxE||0)*ground[0]+(grad.dyE||0)*ground[1],
  dcN=(grad.dtN||0)+(grad.dxN||0)*ground[0]+(grad.dyN||0)*ground[1];
 const inertialCurrent=worldToBody(Q,[dcE,dcN,0]);
 const omegaCrossCurrent=cross(omega,currentBody),
  dcBody=inertialCurrent.map((x,i)=>x-omegaCrossCurrent[i]);
 const currentAccel=[...dcBody,0,0,0];
 const C_RB=coriolisAction(absolute,matVec(par.MRB,0,absolute));
 const C_A=coriolisAction(rel,matVec(par.MA,par.crossSwayYaw,rel));
 const D=rel.map((x,i)=>par.D[i]*x+par.Q[i]*Math.abs(x)*x);
 D[1]+=par.crossDamping*rel[5];D[5]+=par.crossDamping*rel[1];
 const hydro=finiteHydrostaticWrench(state,hull,par),
  sea=waveAt(state.x,state.y,t,inputs.wave||{});
 const euler=toEuler(Q),c=Math.cos(euler.yaw),s=Math.sin(euler.yaw);
 const rollSurface=Math.atan(sea.slopeE*(-s)+sea.slopeN*c),pitchSurface=-Math.atan(sea.slopeE*c+sea.slopeN*s);
 const waveForceBody=worldToBody(Q,[0,0,par.waveExcitationGain*par.Kz*sea.z]);
 // Exogenous regular-wave slope forcing. These are NOT hydrostatic restoring moments
 // or measured diffraction/radiation loads and are excluded from conservation tests.
 const waveTau=[...waveForceBody,par.waveExcitationGain*par.Kroll*Math.sin(rollSurface),par.waveExcitationGain*par.Kpitch*Math.sin(pitchSurface),0];
 const F=inputs.force||{},actuator=[F.X||0,F.Y||0,F.Z||0,F.K||0,F.M||0,F.N||0];
 const RBcurrent=matVec(par.MRB,0,currentAccel);
 const rhs=rel.map((x,i)=>actuator[i]+hydro.tau[i]+waveTau[i]-D[i]-C_RB[i]-C_A[i]-RBcurrent[i]);
 const det=par.M[1]*par.M[5]-par.crossSwayYaw**2;
 const dotNu=rhs.map((x,i)=>x/par.M[i]);
 dotNu[1]=(par.M[5]*rhs[1]-par.crossSwayYaw*rhs[5])/det;
 dotNu[5]=(par.M[1]*rhs[5]-par.crossSwayYaw*rhs[1])/det;
 const Ek=dot(rel,matVec(par.M,par.crossSwayYaw,rel))/2;
 return {dotNu,ground,hydro,sea,waveTau,actuator,rel,absolute,currentBody,rollSurface,pitchSurface,
  kinetic_J:Ek,totalEnergy_J:Ek+hydro.potential_J,dampingPower_W:dot(rel,D),
  generalizedExternalPower_W:dot(rel,actuator)+dot(rel,waveTau)};
}
function step(boat,inputs,dt){
 if(!(dt>0&&dt<=.05))throw Error('6-DOF dt must be within (0,0.05] s');
 const hull=inputs.hull,par=modelParams(hull),time=inputs.time||0;
 const Q=boat.attitude||fromEuler(boat.roll||0,boat.pitch||0,boat.psi||0);
 const start={...boat,attitude:Q};
 const a=evaluateDynamics(start,inputs,par,hull,time),nu=a.rel;
 const midNu=nu.map((x,i)=>x+dt*a.dotNu[i]/2);
 const midQ=advanceQuaternion(Q,midNu.slice(3),dt/2);
 const mid={...start,x:start.x+dt*a.ground[0]/2,y:start.y+dt*a.ground[1]/2,z:(start.z||0)+dt*a.ground[2]/2,
  u:midNu[0],v:midNu[1],w:midNu[2],p:midNu[3],q:midNu[4],r:midNu[5],attitude:midQ};
 // Approximate current at the midpoint from the prescribed local material
 // derivatives so nonuniform-field kinematics are not evaluated at t_n only.
 const grad=inputs.currentDerivatives||{},cv=currentAtState(start,inputs);
 const currRate=[(grad.dtE||0)+(grad.dxE||0)*a.ground[0]+(grad.dyE||0)*a.ground[1],
                 (grad.dtN||0)+(grad.dxN||0)*a.ground[0]+(grad.dyN||0)*a.ground[1]];
 const midpointInputs={...inputs,current:{x:cv[0]+dt/2*currRate[0],y:cv[1]+dt/2*currRate[1]}};
 const b=evaluateDynamics(mid,midpointInputs,par,hull,time+dt/2),next=nu.map((x,i)=>x+dt*b.dotNu[i]);
 const limits=[[-3.5,5],[-3,3],[-2.5,2.5],[-2,2],[-2,2],[-2,2]];
 for(let i=0;i<6;i++)if(!Number.isFinite(next[i])||next[i]<limits[i][0]||next[i]>limits[i][1]){
  boat.limitEvents=(boat.limitEvents||0)+1;boat.physicsValidity='OUT_OF_ENVELOPE_CLAMPED';
  next[i]=Number.isFinite(next[i])?clamp(next[i],limits[i][0],limits[i][1]):0;
 }
 const omegaMid=nu.slice(3).map((x,i)=>(x+next[i+3])/2),orientation=advanceQuaternion(Q,omegaMid,dt),ea=toEuler(orientation);
 const vground=b.ground;
 boat.u=next[0];boat.v=next[1];boat.w=next[2];boat.p=next[3];boat.q=next[4];boat.r=next[5];
 boat.attitude=orientation;boat.roll=ea.roll;boat.pitch=ea.pitch;boat.psi=ea.yaw;
 boat.x+=vground[0]*dt;boat.y+=vground[1]*dt;boat.z=(boat.z||0)+vground[2]*dt;
 boat.dist=(boat.dist||0)+Math.hypot(vground[0],vground[1])*dt;
 const end=evaluateDynamics(boat,{...inputs,current:{x:cv[0]+dt*currRate[0],y:cv[1]+dt*currRate[1]}},par,hull,time+dt);
 boat.waveElevation=end.sea.z;boat.waveHeight=Number(inputs.wave?.height)||0;
 boat.rollEquilibrium=end.rollSurface;boat.pitchEquilibrium=end.pitchSurface;
 boat.heaveRestoring=end.hydro.tau[2];boat.mechanicalEnergy=end.totalEnergy_J;
 boat.displacedVolume=end.hydro.displacedVolume_m3;
 boat.waveModelLimit=end.sea.steepness>.12;
 if(!end.hydro.valid){boat.hydrostaticLimitEvents=(boat.hydrostaticLimitEvents||0)+1;boat.physicsValidity='OUT_OF_HYDROSTATIC_ENVELOPE';}
 boat.stabilityWarning=(Math.abs(ea.roll)>35*Math.PI/180||Math.abs(ea.pitch)>30*Math.PI/180||!end.hydro.valid);
 return {eta:[boat.x,boat.y,boat.z,boat.roll,boat.pitch,boat.psi],nu:next,
  wave:end.sea,worldVelocity:vground,accelerations:b.dotNu,hydrostatics:par.hydro,
  finiteHydrostatics:end.hydro,mechanicalEnergy_J:end.totalEnergy_J,
  dampingPower_W:b.dampingPower_W,externalPower_W:b.generalizedExternalPower_W,
  physicsValidity:boat.physicsValidity||'ILLUSTRATIVE_NOT_VALIDATED',
  physicsGuardrailCount:boat.limitEvents||0};
}
return {G,initialize,modelParams,hydrostatics,finiteHydrostaticWrench,waveAt,step,fromEuler,toEuler,bodyToWorld,worldToBody,advanceQuaternion,evaluateDynamics};
});
'''
POWERTRAIN_ENGINE = r'''
/* BLUEVECTOR V8: SIX-AXIS POINT-VELOCITY AND LIMITED BATTERY POWER ACCOUNTING.
 * Educational actuator-disk proxy; not an identified propeller KT/KQ curve.
 * Frame: forward/port/up. Thruster x-axis: forward; port y=+arm,
 * starboard y=-arm, both z=thrusterZ. Local axial velocity is
 * Vx = u + q*z - r*y, i.e. (v + omega cross position)_x.
 * Enforced constraints: positive generalized propulsive work <= shaft power
 * <= motor electrical input; bus power <= voltage-limited and remaining
 * usable chemical energy; cutoff isolates the pack, no 'ghost propulsion'.
 */
(function(root, factory){const api=factory();if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.BluevectorPowertrain=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 const EPS=1e-10, clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
 function parameters(h){const p=h.powertrain||{};
  return {density:p.waterDensity||1025,discArea:Math.PI*Math.pow((p.propellerDiameter||.28)/2,2),
   efficiency:p.shaftEfficiency||.78,inducedFactor:p.inducedFactor||1.15,
   ratedPower:h.maxMotorPower||850,idlePower:p.motorIdlePower||0,
   minBusVoltage:p.minBusVoltage||18,cutoffSoc:p.cutoffSoc??.02};
 }
 function motorPower(thrust,advance,h){const p=parameters(h),T=Math.abs(thrust);
  const useful=Math.max(0,thrust*advance);
  const induced=p.inducedFactor*Math.pow(T,1.5)/Math.sqrt(2*p.density*p.discArea);
  const shaft=useful+induced;
  const electric=T<EPS?0:shaft/p.efficiency+p.idlePower;
  return {electric,useful,induced,shaft,losses:electric-shaft};
 }
 function axialVelocities(boat,h){
  const u=Number(boat.u)||0,q=Number(boat.q)||0,r=Number(boat.r)||0;
  const z=Number(h.sixDof?.thrusterZ??-.26),arm=Number(h.arm);
  return {port:u+q*z-r*arm,star:u+q*z+r*arm,
   pitchMomentArm:z, yawMomentArm:arm};
 }
 function busLimit(h,energy,auxiliary,dt=1/60){
  if(!(dt>0&&dt<=1))throw Error('invalid simulation timestep');
  const p=parameters(h),R=h.batteryResistance,capacity=h.batteryWh;
  if(!(R>0&&capacity>0&&p.minBusVoltage>0&&p.cutoffSoc>=0&&p.cutoffSoc<1))throw Error('invalid battery parameters');
  const soc=clamp(1-energy/capacity,0,1),voc=20.4+4.8*soc;
  const availableWh=Math.max(0,capacity*(soc-p.cutoffSoc));
  const cutoff=availableWh<=EPS;
  const threshold=Math.max(p.minBusVoltage,voc/2);
  const currentVoltageMax=Math.max(0,(voc-threshold)/R);
  const currentEnergyMax=cutoff?0:availableWh*3600/(voc*dt);
  const maxI=Math.min(currentVoltageMax,currentEnergyMax);
  const deliverable=Math.max(0,(voc-R*maxI)*maxI);
  const requestedAuxiliary=Math.max(0,auxiliary||0);
  return {soc,voc,R,cutoff,availableWh,deliverable,auxiliary:Math.min(requestedAuxiliary,deliverable),
    requestedAuxiliary,energyLimitA:currentEnergyMax,voltageLimitA:currentVoltageMax,
    cutoffSoc:p.cutoffSoc};
 }
 function evaluate(portRequested,starRequested,boat,h,auxiliary=0,dt=1/60){
  const adv=axialVelocities(boat,h),p=parameters(h),bus=busLimit(h,boat.energy,auxiliary,dt);
  function zero(){const z=motorPower(0,0,h);
   return {port:0,star:0,portMotor:z,starMotor:z,motorPower:0,power:bus.auxiliary,
    busVoltage:bus.cutoff?0:bus.voc,batteryCurrent:0,batteryHeat:0,chemicalPower:0,
    usefulPower:0,shaftPower:0,generalizedThrustWork:0,
    scale:0,availableBusPower:bus.deliverable,energyEfficiency:0,
    batteryCutoff:bus.cutoff,auxiliarySupplied:bus.auxiliary,auxiliaryRequested:bus.requestedAuxiliary,
    portAdvance:adv.port,starAdvance:adv.star};
  }
  if(bus.cutoff)return zero();
  function derateMotor(requested,velocity){
   if(motorPower(requested,velocity,h).electric<=p.ratedPower)return requested;
   let lo=0,hi=1;
   for(let j=0;j<45;j++){const mid=(lo+hi)/2;
    if(motorPower(requested*mid,velocity,h).electric>p.ratedPower)hi=mid;else lo=mid;}
   return requested*lo;
  }
  const ratedPort=derateMotor(portRequested,adv.port),ratedStar=derateMotor(starRequested,adv.star);
  const loads=a=>{const port=motorPower(ratedPort*a,adv.port,h),star=motorPower(ratedStar*a,adv.star,h);
   return {port,star,total:port.electric+star.electric+bus.auxiliary};};
  let scale=1;
  if(loads(1).total>bus.deliverable){let lo=0,hi=1;
   for(let j=0;j<45;j++){const mid=(lo+hi)/2;if(loads(mid).total>bus.deliverable)hi=mid;else lo=mid;}
   scale=lo;
  }
  const x=loads(scale);
  const disc=bus.voc*bus.voc-4*bus.R*x.total;
  if(disc<-1e-5)throw Error('electrical power exceeds physical pack limit');
  const current=(bus.voc-Math.sqrt(Math.max(0,disc)))/(2*bus.R);
  const dcVoltage=bus.voc-current*bus.R;
  const batteryHeat=current*current*bus.R;
  const chemicalPower=x.total+batteryHeat;
  const port=ratedPort*scale,star=ratedStar*scale;
  const generalizedThrustWork=(port+star)*(Number(boat.u)||0)+
    (h.sixDof?.thrusterZ??-.26)*(port+star)*(Number(boat.q)||0)+
    h.arm*(star-port)*(Number(boat.r)||0);
  return {port,star,portMotor:x.port,starMotor:x.star,
   motorPower:x.port.electric+x.star.electric,power:x.total,busVoltage:dcVoltage,
   batteryCurrent:current,batteryHeat,chemicalPower,
   usefulPower:x.port.useful+x.star.useful,shaftPower:x.port.shaft+x.star.shaft,
   generalizedThrustWork,portAdvance:adv.port,starAdvance:adv.star,
   scale:Math.min(scale,
    Math.abs(portRequested)>EPS?Math.abs(port/portRequested):1,
    Math.abs(starRequested)>EPS?Math.abs(star/starRequested):1),
   availableBusPower:bus.deliverable,energyEfficiency:x.total>0?(x.port.useful+x.star.useful)/x.total:0,
   batteryCutoff:false,auxiliarySupplied:bus.auxiliary,auxiliaryRequested:bus.requestedAuxiliary};
 }
 return {parameters,motorPower,axialVelocities,busLimit,evaluate};
});
'''

# -----------------------------------------------------------------------------
# PURE PYTHON ENGINEERING / VALIDATION LAYER
# These functions do not import or invoke Streamlit; the analysis UI comes later.
# -----------------------------------------------------------------------------
# Offline engineering algorithms originally stored in marine_engineering.py.
def proxy_hydrostatics(hull: dict = HULL) -> dict:
    """Identical idealized trapezoidal-pontoon hydrostatic proxy used by JS.

    Not a verified underwater mesh or nonlinear righting-arm curve.
    """
    g = hull["sixDof"]["hydroGeometry"]
    length, width, off = (g["waterlineLength"], g["pontoonWidth"], g["pontoonCenterOffset"])
    cb = g["blockCoefficient"]
    if not (0 < length <= hull["length"] and 0 < width and .5 < cb <= 1
            and off >= 0 and off + width / 2 <= hull["beam"] / 2):
        raise ValueError("invalid trapezoidal pontoon geometry")
    volume = hull["mass"] / 1025.0
    area = 2 * length * width
    draft = volume / (area * cb)
    it = 2 * (length * width**3 / 12 + length * width * off**2)
    il = 2 * (width * length**3 / 12)
    cb_depth = draft * (4 * cb - 1) / (6 * cb)
    bg = g["cgHeightAboveWaterline"] + cb_depth
    gmt, gml = it / volume - bg, il / volume - bg
    if min(gmt, gml) <= 0:
        raise ValueError("negative metacentric height")
    return {"displacement_m3": volume, "draft_m": draft,
            "waterplane_area_m2": area, "GMt_m": gmt, "GMl_m": gml,
            "center_buoyancy_depth_m": cb_depth, "bottom_width_ratio": 2 * cb - 1,
            "heave_stiffness_Npm": 1025 * 9.80665 * area,
            "roll_stiffness_Nm": hull["mass"] * 9.80665 * gmt,
            "pitch_stiffness_Nm": hull["mass"] * 9.80665 * gml}


def motor_electrical_power(thrust: float, advance: float, hull: dict = HULL) -> float:
    """Electrical input >= positive axial work + nonnegative induced proxy."""
    p = hull["powertrain"]
    T = abs(thrust)
    if T < 1e-10:
        return 0.0
    area = np.pi * (p["propellerDiameter"] / 2)**2
    useful = max(0.0, thrust * advance)
    induced = p["inducedFactor"] * T**1.5 / np.sqrt(2 * p["waterDensity"] * area)
    return (useful + induced) / p["shaftEfficiency"] + p["motorIdlePower"]


def evaluate_powertrain(port_requested: float, star_requested: float, u: float, r: float,
                        energy_Wh: float, auxiliary_W: float, hull: dict = HULL,
                        q: float = 0., dt: float = 1/60) -> dict:
    """Reference for JS energy accounting. q is pitch rate; port y is positive."""
    z = hull["sixDof"]["thrusterZ"]
    vp, vs = u + q*z - r*hull["arm"], u + q*z + r*hull["arm"]
    soc = float(np.clip(1 - energy_Wh/hull["batteryWh"], 0, 1))
    voc = 20.4 + 4.8 * soc
    R = hull["batteryResistance"]
    reserve = hull["batteryWh"] * max(0., soc - hull["powertrain"].get("cutoffSoc",.02))
    cutoff = reserve <= 1e-10
    floor = max(hull["powertrain"]["minBusVoltage"], voc/2)
    imax = min(max(0., (voc-floor)/R), reserve*3600/(voc*dt)) if not cutoff else 0.
    available = max(0., (voc - R*imax)*imax)
    auxiliary = min(max(0., auxiliary_W), available)
    if cutoff:
        return {"port_N": 0., "star_N": 0., "power_W": 0., "chemical_W": 0.,
                "battery_loss_W": 0., "useful_W": 0., "motor_power_W": 0.,
                "shaft_W":0., "body_work_W":0., "bus_V": 0., "derate": 0.,
                "battery_cutoff": True, "available_W":0.}
    def motor_limit(requested, advance):
        if motor_electrical_power(requested,advance,hull)<=hull["maxMotorPower"]:
            return requested
        lo,hi=0.,1.
        for _ in range(45):
            mid=(lo+hi)/2
            if motor_electrical_power(requested*mid,advance,hull)>hull["maxMotorPower"]: hi=mid
            else: lo=mid
        return requested*lo
    rated_port = motor_limit(port_requested,vp)
    rated_star = motor_limit(star_requested,vs)
    def loads(scale):
        p1 = motor_electrical_power(rated_port*scale, vp, hull)
        p2 = motor_electrical_power(rated_star*scale, vs, hull)
        return p1,p2,p1+p2+auxiliary
    scale = 1.
    if loads(scale)[2] > available:
        lo,hi=0.,1.
        for _ in range(45):
            mid=(lo+hi)/2
            if loads(mid)[2]>available: hi=mid
            else: lo=mid
        scale=lo
    p1,p2,total=loads(scale)
    disc=voc*voc-4*R*total
    if disc<-1e-5:raise ValueError("power exceeds electrical capability")
    current=(voc-np.sqrt(max(0.,disc)))/(2*R)
    battery_loss=current**2*R
    actual_port,actual_star = rated_port*scale,rated_star*scale
    shaft=max(0.,actual_port*vp)+max(0.,actual_star*vs)
    area=np.pi*(hull["powertrain"]["propellerDiameter"]/2)**2
    c=hull["powertrain"]["inducedFactor"]/np.sqrt(2*hull["powertrain"]["waterDensity"]*area)
    shaft+=c*(abs(actual_port)**1.5+abs(actual_star)**1.5)
    body_work=(actual_port+actual_star)*u + z*(actual_port+actual_star)*q + hull["arm"]*(actual_star-actual_port)*r
    return {"port_N":actual_port,"star_N":actual_star,"power_W":total,
            "battery_loss_W":battery_loss,"chemical_W":total+battery_loss,
            "useful_W":max(0,actual_port*vp)+max(0,actual_star*vs),"motor_power_W":p1+p2,
            "shaft_W":shaft,"body_work_W":body_work,"bus_V":voc-current*R,
            "derate":min(scale,abs(actual_port/port_requested) if port_requested else 1,
                         abs(actual_star/star_requested) if star_requested else 1),
            "battery_cutoff":False,"available_W":available}


@dataclass
class Scenario:
    mode: str = "advanced"
    maneuver: str = "Acceleration and coast-down"
    duration_s: float = 60.0
    current_e: float = 0.2
    current_n: float = -0.08
    wind_e: float = 0.0
    wind_n: float = 0.0
    wave_amp_N: float = 6.0
    motor_tau: float = 0.45
    seed: int = 2026
    time_step_s: float = 1 / 60


def allocate_thrusters(surge_N: float, yaw_Nm: float, arm: float,
                       max_thrust: float, *, surge_weight: float = 1.0,
                       yaw_weight: float = 1.0,
                       regularization: float = 1e-11) -> tuple[float, float]:
    """Portable bounded least-squares twin-thruster allocation.

    Body x-forward, y-port, z-up; B=[[1,1],[-arm,arm]].
    For the two actuators, the global convex box-constrained optimum is
    interior or on a box edge, so enumerate those candidates analytically.
    """
    if arm <= 0 or max_thrust <= 0:
        raise ValueError("arm and max_thrust must be positive")
    fscale, mscale = 2 * max_thrust, 2 * arm * max_thrust
    w1, w2 = max(0.01, surge_weight), max(0.01, yaw_weight)
    A = np.array([[w1/fscale, w1/fscale],
                  [-w2*arm/mscale, w2*arm/mscale]], dtype=float)
    b = np.array([w1*surge_N/fscale, w2*yaw_Nm/mscale], dtype=float)
    H = A.T @ A + max(0.0, regularization) * np.eye(2)
    g = A.T @ b
    lim = float(max_thrust)
    candidates = []
    x = np.linalg.solve(H, g)
    if np.all(np.abs(x) <= lim):
        candidates.append(x)
    for fixed in (-lim, lim):
        candidates.append(np.array([fixed, np.clip((g[1]-H[1,0]*fixed)/H[1,1], -lim, lim)]))
        candidates.append(np.array([np.clip((g[0]-H[0,1]*fixed)/H[0,0], -lim, lim), fixed]))
    best = min(candidates, key=lambda v: float(np.sum((A@v-b)**2) + max(0,regularization)*np.sum(v*v)))
    return float(best[0]), float(best[1])


def _wrap(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def run_maneuver(sc: Scenario) -> pd.DataFrame:
    """Numerically integrates an idealized USV maneuver with fixed-step Euler integration.

    End state and time are sampled at regular intervals. Wind and planar wave forces
    use the same simplified parameterization as the browser model. Random perturbation
    is seeded and represents experiment-to-experiment variation, not sea-state spectra.
    """
    p = HULL
    if sc.mode not in {"advanced", "baseline"}:
        raise ValueError("mode must be advanced or baseline")
    dt = float(sc.time_step_s)
    if not 0 < dt <= 0.1 or not 1 <= sc.duration_s <= 180:
        raise ValueError("invalid timestep or duration")
    rng = np.random.default_rng(sc.seed)
    # Small, smooth realization-specific parameter deviations, held per run.
    damping_scale = float(np.clip(rng.normal(1, .05), .75, 1.25))
    wind_scale = float(np.clip(rng.normal(1, .08), .7, 1.3))
    x, y, psi, u, v, r, motor_p, motor_s, energy = (21., 48., 0., 0., 0., 0., 0., 0., 0.)
    motor_temp = 20.
    actual_p = actual_s = 0.
    rows = []
    n = int(round(sc.duration_s / dt))
    for step in range(n + 1):
        t = step * dt
        c, sn = np.cos(psi), np.sin(psi)
        flow_e, flow_n = sc.current_e, sc.current_n
        ground_e = c*u - sn*v + flow_e
        ground_n = sn*u + c*v + flow_n
        if step % max(1, round(.25 / dt)) == 0 or step == n:
            rows.append({"t_s": t, "east_m": x, "north_m": y,
                         "yaw_deg_enu": np.degrees(psi),
                         "heading_deg_true": (90-np.degrees(psi)) % 360,
                         "surge_mps": u, "sway_mps": v, "yaw_rate_dps": np.degrees(r),
                         "sog_mps": np.hypot(ground_e, ground_n),
                         "stw_mps": np.hypot(u, v),
                         "port_N": actual_p, "starboard_N": actual_s,
                         "energy_Wh": energy, "motor_temp_C": motor_temp,
                         "mode": sc.mode, "maneuver": sc.maneuver})
        if step == n:
            break
        if sc.maneuver == "Acceleration and coast-down":
            throttle = 0.72 if t < sc.duration_s * .55 else 0.
            steer = 0.
        elif sc.maneuver == "Differential thrust turn":
            throttle = .50
            steer = .30 if t > 3 else 0.
        elif sc.maneuver == "Heading hold":
            target = np.deg2rad(25)  # ENU yaw, 65 degrees true heading
            err = _wrap(target - psi)
            steer = float(np.clip(-1.3*err + .85*r, -.78, .78))
            throttle = float(np.clip(.62*(2.0-u), -.5, .95))
        else:
            raise ValueError("unknown maneuver")
        wanted_p, wanted_s = allocate_thrusters(
            2 * p["maxThrust"] * throttle,
            -2 * p["arm"] * p["maxThrust"] * steer,
            p["arm"], p["maxThrust"]
        )
        alpha = 1 - np.exp(-dt/sc.motor_tau) if sc.mode == "advanced" else 1.0
        motor_p += alpha * (wanted_p - motor_p)
        motor_s += alpha * (wanted_s - motor_s)
        electric = evaluate_powertrain(motor_p, motor_s, u, r, energy, p["hotelPower"], p)
        actual_p,actual_s = electric["port_N"],electric["star_N"]
        wind_relative_x = sc.wind_e*wind_scale - ground_e
        wind_relative_y = sc.wind_n*wind_scale - ground_n
        airmag = np.hypot(wind_relative_x, wind_relative_y)
        wind_f_e = .60 * airmag * wind_relative_x
        wind_f_n = .60 * airmag * wind_relative_y
        wfX = c * wind_f_e + sn * wind_f_n
        wfY = -sn * wind_f_e + c * wind_f_n
        wfN = .12 * wfY
        phase = 1.4*t+.065*x+.04*y
        f_e = .25*sc.wave_amp_N*np.cos(phase*1.15)
        f_n = sc.wave_amp_N*np.sin(phase)
        waveX = c*f_e+sn*f_n
        waveY = -sn*f_e+c*f_n
        waveN = .12*sc.wave_amp_N*np.sin(phase+.7)
        X = actual_p + actual_s - damping_scale*(p["Xu"]*u + p["Xuu"]*abs(u)*u) + wfX + waveX
        Y = -damping_scale*(p["Yv"]*v+p["Yvv"]*abs(v)*v)+wfY+waveY
        N = p["arm"]*(actual_s-actual_p)-damping_scale*(p["Nr"]*r+p["Nrr"]*abs(r)*r)+wfN+waveN
        if sc.mode == "advanced":
            h = p["m22"]*v+p["m23"]*r
            rhs_u = X+h*r
            rhs_v = Y-p["m11"]*u*r-p["Dvr"]*r
            rhs_n = N-h*u+p["m11"]*u*v-p["Dvr"]*v
            det = p["m22"]*p["m33"]-p["m23"]**2
            du = rhs_u/p["m11"]
            dv = (p["m33"]*rhs_v-p["m23"]*rhs_n)/det
            dr = (p["m22"]*rhs_n-p["m23"]*rhs_v)/det
        else:
            du = (X+p["m22"]*v*r)/p["m11"]
            dv = (Y-p["m11"]*u*r)/p["m22"]
            dr = (N-(p["m22"]-p["m11"])*u*v)/p["m33"]
        u = float(np.clip(u+du*dt, -3.5, 5))
        v = float(np.clip(v+dv*dt, -2.5, 2.5))
        r = float(np.clip(r+dr*dt, -1.5, 1.5))
        psi = _wrap(psi+r*dt)
        # Match browser semi-implicit Euler: new body state determines displacement.
        c2, sn2 = np.cos(psi), np.sin(psi)
        ground_e = c2*u-sn2*v + flow_e
        ground_n = sn2*u+c2*v + flow_n
        x += ground_e*dt
        y += ground_n*dt
        motor_temp += ((electric["motor_power_W"] * (1-p["powertrain"]["shaftEfficiency"]))
                       -p["motorCooling"]*(motor_temp-20))*dt/p["motorThermalCapacity"]
        energy += electric["chemical_W"]*dt/3600
    return pd.DataFrame(rows)


def summarize_runs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    summaries = []
    for k, df in enumerate(dfs, start=1):
        last = df.iloc[-1]
        summaries.append({"trial": k, "end_east_m": last.east_m,
                          "end_north_m": last.north_m,
                          "final_sog_mps": last.sog_mps,
                          "peak_sog_mps": df.sog_mps.max(),
                          "final_heading_deg": last.heading_deg_true,
                          "energy_Wh": last.energy_Wh,
                          "motor_temp_C": last.motor_temp_C})
    return pd.DataFrame(summaries)


def finite_attitude_buoyancy(heave_m: float = 0., roll_rad: float = 0.,
                            pitch_rad: float = 0., yaw_rad: float = 0.,
                            hull: dict = HULL) -> dict:
    """Independent Python reference for the V9 calm-water nonlinear buoyancy.

    Uses the same prismatic trapezoidal-pontoon proxy and separated 4-point
    Gaussian quadrature in the cross-section as the JavaScript live model.
    It is a numerical verification reference, not experimental validation.
    Body axes are X-forward/Y-port/Z-up, all in SI units; attitude ZYX.
    """
    hp = proxy_hydrostatics(hull)
    geom = hull["sixDof"]["hydroGeometry"]
    length, beam, off = geom["waterlineLength"], geom["pontoonWidth"], geom["pontoonCenterOffset"]
    r = hp["bottom_width_ratio"]
    draft = hp["draft_m"]
    freeboard = geom.get("pontoonFreeboard", .19)
    cg_z = geom["cgHeightAboveWaterline"]
    # Independent attitude matrix, Rz(yaw) Ry(pitch) Rx(roll).
    sr, cr = np.sin(roll_rad), np.cos(roll_rad)
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    sy, cy = np.sin(yaw_rad), np.cos(yaw_rad)
    R = np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                  [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                  [-sp,cp*sr,cp*cr]],dtype=float)
    if R[2, 2] < .12:
        raise ValueError("pontoon hydrostatics valid only for sufficiently upright attitudes")
    xgrid = (np.arange(24)+.5)*(length/24)-length/2
    nodes, weights = np.polynomial.legendre.leggauss(4)
    V, mx, my, mz, worldz_sum = 0.,0.,0.,0.,0.
    for side in (-1, 1):
        for x in xgrid:
            for lo,hi in [(0.,r*beam/2),(r*beam/2,beam/2)]:
                for node,wt in zip(nodes,weights):
                    yl=(lo+hi)/2+(hi-lo)*node/2
                    dyw=(hi-lo)*wt/2*(length/24)
                    base=-draft if yl<=r*beam/2+1e-15 else -draft+draft*(2*yl/beam-r)/(1-r)
                    for sg in (-1, 1):
                        y=side*off+sg*yl
                        zcut=(-heave_m-R[2,0]*x-R[2,1]*y)/R[2,2]
                        top=min(freeboard,zcut)
                        if top<=base:
                            continue
                        dv=(top-base)*dyw
                        zm=(top**2-base**2)*dyw/2
                        V+=dv
                        mx+=x*dv;my+=y*dv;mz+=zm
                        worldz_sum+=heave_m*dv+R[2,0]*x*dv+R[2,1]*y*dv+R[2,2]*zm
    body_buoyancy=R.T@np.array([0.,0.,1025.*9.80665*V])
    body_weight=R.T@np.array([0.,0.,-hull["mass"]*9.80665])
    cb=np.array([mx/V,my/V,mz/V]) if V>0 else np.zeros(3)
    cg=np.array([0.,0.,cg_z])
    wrench=np.r_[body_buoyancy+body_weight,
                 np.cross(cb,body_buoyancy)+np.cross(cg,body_weight)]
    U=hull["mass"]*9.80665*(heave_m+R[2,2]*cg_z)-1025.*9.80665*worldz_sum
    return {"displaced_volume_m3":V,"center_buoyancy_body_m":cb,
            "wrench_body":wrench,"potential_J":U}


def illustrative_linear_rao(periods_s, hull: dict = HULL) -> pd.DataFrame:
    """Small-amplitude *model* frequency response, NOT measured seakeeping RAOs.

    Assumes the V9 surrogate wave excitation Fz=gain*Kz*eta and
    Mroll/pitch=gain*Kroll/pitch*k*eta. Frequency-dependent radiation
    damping, diffraction, encounter-frequency correction, and forward-speed
    hydrodynamic effects are NOT present.
    """
    hp=proxy_hydrostatics(hull)
    periods=np.asarray(periods_s,dtype=float)
    if np.any(~np.isfinite(periods)) or np.any(periods<=0):
        raise ValueError('wave periods must be positive and finite')
    omega=2*np.pi/periods
    k=omega**2/9.80665
    d=hull['sixDof']
    gain=d.get('waveExcitationGain',.5)
    def response(mass,damping,kst,force_amp):
        return np.abs(force_amp/(kst-mass*omega**2+1j*damping*omega))
    heave=response(d['heaveMass'],d['heaveDamping'],hp['heave_stiffness_Npm'],gain*hp['heave_stiffness_Npm'])
    roll=response(d['rollInertia'],d['rollDamping'],hp['roll_stiffness_Nm'],gain*hp['roll_stiffness_Nm']*k)*180/np.pi
    pitch=response(d['pitchInertia'],d['pitchDamping'],hp['pitch_stiffness_Nm'],gain*hp['pitch_stiffness_Nm']*k)*180/np.pi
    return pd.DataFrame({'period_s':periods,'heave_rao_m_per_m':heave,
                         'roll_rao_deg_per_m':roll,'pitch_rao_deg_per_m':pitch})

# Engineering Analysis formerly under pages/, now rendered within this app.
# STREAMLIT ANALYSIS ADAPTER. Computation utilities above can move independently.
def render_engineering_analysis():
    st.title("BLUEVECTOR | Engineering Analysis")
    st.caption("Offline research and imported telemetry replay · Shared nominal hull coefficients · Not a validated vessel model")
    replay, bench, monte, hydro_check = st.tabs(["Mission Telemetry Replay", "Hydrodynamic Benchmarks", "Monte Carlo Experiments", "6-DOF Integrity"])
    with replay:
        st.subheader("Analyze an exported BLUEVECTOR telemetry CSV")
        file = st.file_uploader("Import bluevector_usv_telemetry.csv", type="csv", key="telemetry")
        if file:
            try:
                df = pd.read_csv(file)
                required = {"t", "x_e_m", "y_n_m", "sog_mps", "heading_deg", "power_W", "energy_Wh"}
                if not required.issubset(df.columns):
                    raise ValueError("CSV is missing expected BLUEVECTOR telemetry columns: " + ", ".join(sorted(required-set(df.columns))))
                df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)
                numeric = [c for c in ["t", "x_e_m", "y_n_m", "sog_mps", "power_W", "energy_Wh", "nav_error_m", "xte_m", "yaw_dps",
                                       "heave_z_m", "heave_w_mps", "roll_deg", "pitch_deg", "roll_rate_dps", "pitch_rate_dps", "wave_surface_m"] if c in df]
                for c in numeric:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["t", "x_e_m", "y_n_m", "sog_mps"])
                if len(df) < 2:
                    raise ValueError("At least two valid telemetry samples required")
                cols = st.columns(4)
                cols[0].metric("Recorded duration", f"{df.t.max()-df.t.min():.1f} s")
                cols[1].metric("Peak speed", f"{df.sog_mps.max():.2f} m/s")
                cols[2].metric("Energy used", f"{df.energy_Wh.iloc[-1]-df.energy_Wh.iloc[0]:.2f} Wh")
                if "nav_error_m" in df:
                    rmse = np.sqrt(np.nanmean(np.square(df.nav_error_m)))
                    cols[3].metric("Navigation RMSE", f"{rmse:.2f} m")
                else:
                    cols[3].metric("Telemetry samples", len(df))
                left, right = st.columns(2)
                with left:
                    fig = px.line(df, x="t", y=[c for c in ["sog_mps", "speed_target_mps"] if c in df],
                                  labels={"t":"Mission time (s)","value":"Speed (m/s)","variable":"Series"}, title="Speed Response")
                    fig.update_layout(template="plotly_dark", legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    fig = px.line(df, x="x_e_m", y="y_n_m", title="Ground Track in Local ENU")
                    fig.update_layout(template="plotly_dark", xaxis_title="East (m)", yaxis_title="North (m)", yaxis_scaleanchor="x")
                    st.plotly_chart(fig, use_container_width=True)
                avail = [c for c in ["nav_error_m", "sigma_pos_m", "xte_m", "yaw_dps", "power_W", "dc_voltage_V", "motor_temp_C"] if c in df]
                if avail:
                    chosen = st.multiselect("Additional engineering traces", avail, default=avail[:2])
                    if chosen:
                        fig = px.line(df, x="t", y=chosen, title="Recorded Engineering Parameters")
                        fig.update_layout(template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Mixed units (see legends)")
                        st.plotly_chart(fig, use_container_width=True)
                if {"heave_z_m", "roll_deg", "pitch_deg"}.issubset(df.columns):
                    st.subheader("Recorded six-axis vessel motion")
                    if "physics_mode" in df:
                        st.caption("Recorded model: " + ", ".join(df.physics_mode.astype(str).unique()))
                    a, b = st.columns(2)
                    with a:
                        cols = [c for c in ["heave_z_m", "wave_surface_m"] if c in df]
                        fig = px.line(df, x="t", y=cols, title="Heave vs. regular wave surface")
                        fig.update_layout(template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Local Up (m)")
                        st.plotly_chart(fig, use_container_width=True)
                    with b:
                        fig = px.line(df, x="t", y=["roll_deg", "pitch_deg"], title="Six-DOF attitude")
                        fig.update_layout(template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Angle (°)")
                        st.plotly_chart(fig, use_container_width=True)
                    metrics = st.columns(3)
                    metrics[0].metric("Peak |heave|", f"{df.heave_z_m.abs().max():.3f} m")
                    metrics[1].metric("Peak |roll|", f"{df.roll_deg.abs().max():.2f}°")
                    metrics[2].metric("Peak |pitch|", f"{df.pitch_deg.abs().max():.2f}°")
                    st.caption("These are integrated browser-model motions from illustrative, uncalibrated parameters, not verified sea-trial results.")
                if "yaw_dps" in df and len(df) > 12:
                    tvals = df.t.to_numpy(dtype=float)
                    dt = np.median(np.diff(tvals))
                    if dt > 0 and np.max(np.abs(np.diff(tvals)-dt)) < dt*.1:
                        f, pxx = welch(df.yaw_dps.ffill().to_numpy(), fs=1/dt, nperseg=min(128,len(df)))
                        st.caption(f"Dominant non-DC yaw-rate spectral component (illustrative): {f[np.argmax(pxx[1:])+1]:.3f} Hz" if len(f)>1 else "Insufficient spectral bins")
                st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "bluevector_replay_cleaned.csv", "text/csv")
            except Exception as exc:
                st.error(f"Unable to process telemetry: {exc}")
        else:
            st.info("Run BLUEVECTOR on the main page, export CSV telemetry, then upload it here to analyze recorded mission data.")

    with bench:
        st.subheader("Baseline versus coupled 3-DOF response")
        with st.form("bench_settings"):
            maneuver = st.selectbox("Maneuver", ["Acceleration and coast-down", "Differential thrust turn", "Heading hold"])
            dur = st.slider("Duration (s)", 20, 120, 60, 10)
            curr = st.slider("East current (m/s)", -0.6, 0.6, 0.2, 0.05)
            wave = st.slider("Planar wave force amplitude (N)", 0., 20., 6., 1.)
            tau = st.slider("Advanced motor time constant (s)", .1, 1.2, .45, .05)
            run = st.form_submit_button("Run paired benchmark")
        if run:
            base = Scenario(mode="baseline", maneuver=maneuver, duration_s=dur, current_e=curr, wave_amp_N=wave, motor_tau=tau)
            adv = Scenario(mode="advanced", maneuver=maneuver, duration_s=dur, current_e=curr, wave_amp_N=wave, motor_tau=tau)
            bdf, adf = run_maneuver(base), run_maneuver(adv)
            comparisons = pd.concat([bdf,adf],ignore_index=True)
            for key,title,ylabel in [("sog_mps","Speed over Ground","Speed (m/s)"),("yaw_rate_dps","Yaw-rate Response","Yaw rate (°/s)"),("energy_Wh","Electrical Energy Use","Energy (Wh)")]:
                fig=px.line(comparisons,x="t_s",y=key,color="mode",title=title)
                fig.update_layout(template="plotly_dark",xaxis_title="Time (s)",yaxis_title=ylabel)
                st.plotly_chart(fig,use_container_width=True)
            st.dataframe(pd.DataFrame({"Model":["Baseline","Advanced"],
                                      "End speed (m/s)":[bdf.sog_mps.iloc[-1],adf.sog_mps.iloc[-1]],
                                      "Energy used (Wh)":[bdf.energy_Wh.iloc[-1],adf.energy_Wh.iloc[-1]],
                                      "Final heading (°)":[bdf.heading_deg_true.iloc[-1],adf.heading_deg_true.iloc[-1]]}).round(3),hide_index=True)
            st.caption("The benchmark uses the shared nominal hull coefficients and a separate, seeded offline integrator. It is not a replay of the live browser vessel state.")

    with monte:
        st.subheader("Repeatable uncertainty experiments")
        with st.form("mc_settings"):
            mc_runs = st.slider("Trials", 5, 50, 20, 5)
            mc_duration = st.slider("Trial duration (s)", 20, 100, 50, 10)
            mc_maneuver = st.selectbox("Maneuver", ["Differential thrust turn", "Acceleration and coast-down", "Heading hold"], key="mc_man")
            mc_seed = st.number_input("Seed", min_value=1, max_value=9999999, value=2026)
            mc_model = st.selectbox("Physics mode", ["advanced", "baseline"])
            execute = st.form_submit_button("Run Monte Carlo")
        if execute:
            data = [run_maneuver(Scenario(mode=mc_model, maneuver=mc_maneuver,
                       duration_s=mc_duration,seed=int(mc_seed)+i,time_step_s=1/30)) for i in range(mc_runs)]
            summary = summarize_runs(data)
            mean=float(summary.peak_sog_mps.mean())
            ci = stats.t.interval(.95,len(summary)-1, loc=mean, scale=stats.sem(summary.peak_sog_mps))
            if not np.all(np.isfinite(ci)):
                ci=(mean,mean)
            c1,c2,c3=st.columns(3)
            c1.metric("Mean peak SOG",f"{mean:.2f} m/s")
            c2.metric("95% CI of sample mean", f"[{ci[0]:.2f}, {ci[1]:.2f}] m/s")
            c3.metric("End-point spread (E)",f"{summary.end_east_m.std():.2f} m")
            fig=go.Figure()
            for d in data:
                fig.add_trace(go.Scatter(x=d.east_m,y=d.north_m,mode="lines",line=dict(width=1),opacity=.55,showlegend=False))
            fig.update_layout(template="plotly_dark",title="Simulated Ground-Track Ensemble",xaxis_title="East (m)",yaxis_title="North (m)",yaxis_scaleanchor="x")
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(summary.round(3),hide_index=True)
            st.download_button("Download experiment summary",summary.to_csv(index=False).encode(),"bluevector_monte_carlo.csv","text/csv")
            st.caption("Trial variability represents simplified coefficient and wind-scaling uncertainty, not calibrated uncertainty bounds or environmental certification.")


    with hydro_check:
        st.subheader("Finite-attitude hydrostatic restoring response")
        st.caption("Computed from the V9 submerged-volume geometry with a consistent gravity/buoyancy potential. Assumptions: 4 m illustrative pontoon hull, seawater 1025 kg/m³, no validated mass distribution.")
        max_angle=st.slider("Attitude sweep (±degrees)",3,20,12,1)
        angles=np.linspace(-max_angle,max_angle,55)
        hdata=[]
        for axis in ['roll','pitch']:
            for a in angles:
                args={'roll_rad':np.deg2rad(a)} if axis=='roll' else {'pitch_rad':np.deg2rad(a)}
                v=finite_attitude_buoyancy(**args)
                hdata.append({'axis':axis,'angle_deg':a,
                             'restoring_moment_Nm':v['wrench_body'][3 if axis=='roll' else 4],
                             'displaced_volume_m3':v['displaced_volume_m3']})
        hd=pd.DataFrame(hdata)
        chart=px.line(hd,x='angle_deg',y='restoring_moment_Nm',color='axis',title='Finite-Attitude Restoring Moment (Body Coordinates)')
        chart.update_layout(template='plotly_dark',xaxis_title='Attitude (degrees)',yaxis_title='Restoring moment (N·m)')
        st.plotly_chart(chart,use_container_width=True)
        geom=HULL['sixDof']['hydroGeometry']
        hp=proxy_hydrostatics(HULL)
        m1,m2,m3=st.columns(3)
        m1.metric('Nominal draft',f"{hp['draft_m']:.3f} m")
        m2.metric('Transverse GM (proxy)',f"{hp['GMt_m']:.2f} m")
        m3.metric('Longitudinal GM (proxy)',f"{hp['GMl_m']:.2f} m")
        st.warning('This curve is calculated for an assumed geometry. Emergence, high heel, freeboard immersion, and mass distribution remain uncalibrated. It cannot establish seaworthiness or capsize limits.')
        st.subheader('Linearized model frequency response')
        periods=np.linspace(1.5,8.,110)
        rao=illustrative_linear_rao(periods)
        which=st.selectbox('Response metric',['heave_rao_m_per_m','roll_rao_deg_per_m','pitch_rao_deg_per_m'])
        fig=px.line(rao,x='period_s',y=which,title='Analytical Linear-Model Response (NOT a Measured RAO)')
        fig.update_layout(template='plotly_dark',xaxis_title='Regular-wave period (s)',yaxis_title=which)
        reference=st.file_uploader('Optional measured reference data (CSV: period_s + selected response column)',type='csv',key='rao_reference')
        if reference:
            try:
                ref=pd.read_csv(reference)
                if not {'period_s',which}.issubset(ref.columns):raise ValueError(f'CSV needs period_s and {which}')
                x=pd.to_numeric(ref.period_s,errors='coerce').to_numpy()
                y=pd.to_numeric(ref[which],errors='coerce').to_numpy()
                good=np.isfinite(x)&np.isfinite(y)&(x>0)
                x,y=x[good],y[good]
                if len(x)<3:raise ValueError('At least 3 finite reference points are required')
                pred=np.interp(x,rao.period_s.to_numpy(),rao[which].to_numpy())
                rmse=float(np.sqrt(np.mean((pred-y)**2)))
                fig.add_trace(go.Scatter(x=x,y=y,mode='markers',name='External reference (user provided)'))
                st.metric('Model vs reference RMSE',f'{rmse:.4g}')
                st.caption('A comparison against external measurements is informative but cannot establish whole-envelope validation. Confirm vessel identity, amplitude normalization, units, frequency/encounter convention, and sea conditions.')
            except Exception as e:
                st.error(f'Invalid comparison file: {e}')
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Download model response CSV',rao.to_csv(index=False).encode(),'bluevector_linear_model_rao.csv','text/csv')
        st.caption('This analytical response ignores frequency-dependent radiation/diffraction, forward-speed effects and irregular-wave statistics. No external seakeeping coefficients are embedded.')

    st.divider()
    st.caption("Model fidelity: live simulator offers selectable uncalibrated 3-DOF and 6-DOF; this page's independent Python maneuver benchmarks remain 3-DOF only. No measured hull coefficients, certified navigation, or COLREG compliance.")




# -----------------------------------------------------------------------------
# STREAMLIT ADAPTER: collects inputs but does not implement vessel physics.
# A future web or desktop frontend may replace these functions without changing
# the embedded numerical engines or the public browser-state contract.
# -----------------------------------------------------------------------------
def render_streamlit_controls() -> tuple[str, dict[str, Any]]:
    with st.sidebar:
        workspace = st.radio("Workspace", ["Live Simulation", "Engineering Analysis"], index=0, horizontal=False)
        st.divider()
        st.header("⚙️ BLUEVECTOR Scenario")
        st.caption("Changing sidebar parameters restarts the live browser simulation.")
        physics_mode = st.selectbox("Dynamics model", ["Advanced coupled 3-DOF", "Advanced illustrative 6-DOF", "Baseline 3-DOF"])
        sixdof_active = physics_mode == "Advanced illustrative 6-DOF"
        nav_filter = st.selectbox("Navigation estimator", ["Extended Kalman Filter", "Legacy complementary"])
        motor_tau = st.slider("Motor response time (s)", 0.05, 1.5, 0.45, 0.05)
        current_field = st.toggle("Nonuniform current field", value=False)
        gnss_outlier = st.slider("GNSS outlier probability (%)", 0, 15, 2, 1)
        gyro_bias = st.slider("Gyro true bias (°/s)", 0.0, 0.8, 0.15, 0.05)
        control = st.selectbox("Mobile helm", ["Joystick", "D-pad"], index=0)
        current_e = st.slider("Eastward current (m/s)", -0.8, 0.8, 0.20, 0.05)
        current_n = st.slider("Northward current (m/s)", -0.8, 0.8, -0.08, 0.05)
        wind_speed = st.slider("Wind speed (m/s)", 0.0, 12.0, 4.0, 0.5)
        wind_from = st.slider("Wind FROM true bearing (°)", 0, 359, 270, 1)
        wave_amp = st.slider("Horizontal wave-disturbance force (N)", 0, 20, 6, 1)
        st.subheader("6-DOF regular-wave forcing")
        st.caption("Applied only to 6-DOF. Finite-attitude twin-pontoon buoyancy and externally forced regular waves remain ILLUSTRATIVE. Wave response is not a validated RAO or diffraction model. Excessive heel, immersion, and steep waves are flagged.")
        wave_height = st.slider("Regular wave height, crest-to-trough (m)", 0.0, 0.6, 0.18, 0.02, disabled=not sixdof_active)
        wave_period = st.slider("Wave period (s)", 2.0, 7.0, 3.5, 0.25, disabled=not sixdof_active)
        wave_bearing = st.slider("Wave travel bearing (° true)", 0, 359, 45, 5, disabled=not sixdof_active)
        wave_steepness = (wave_height / 2) * ((2 * 3.141592653589793 / wave_period)**2) / 9.80665
        if sixdof_active and wave_steepness > 0.12:
            st.warning(f"Outside intended linear-wave envelope: ka={wave_steepness:.3f} > 0.12. Wave loads are illustrative, not suitable for motion prediction.")
        gnss_noise = st.slider("GNSS one-axis noise σ (m)", 0.2, 5.0, 1.2, 0.2)
        max_thrust = st.slider("Max thrust per motor (N)", 80, 240, 160, 10)
        show_contact = st.toggle("Enable moving traffic contact", value=True)
        seed = st.number_input("Scenario random seed", min_value=1, max_value=999999, value=2026)
        st.divider()
        st.caption(
            "Unvalidated 4 m twin-thruster demonstration hull. Optional 6-DOF model "
            "adds six-axis motion, volume-integrated pontoon buoyancy, and illustrative regular-wave excitation. "
            "All three physics modes are illustrative, not calibrated or COLREG-compliant."
        )

    cfg = {
        "control": "joystick" if control == "Joystick" else "dpad",
        "currentE": float(current_e),
        "currentN": float(current_n),
        "windSpeed": float(wind_speed),
        "windFrom": float(wind_from),
        "waveAmp": float(wave_amp),
        "gnssSigma": float(gnss_noise),
        "maxThrust": float(max_thrust),
        "traffic": bool(show_contact),
        "seed": int(seed),
        "physicsMode": "6dof" if sixdof_active else ("advanced" if physics_mode.startswith("Advanced") else "baseline"),
        "waveHeight": float(wave_height),
        "wavePeriod": float(wave_period),
        "waveBearing": int(wave_bearing),
        "navFilter": "ekf" if nav_filter.startswith("Extended") else "legacy",
        "motorTau": float(motor_tau),
        "currentField": bool(current_field),
        "gnssOutlier": float(gnss_outlier) / 100,
        "gyroBiasTrue": float(gyro_bias) * 3.141592653589793 / 180,
        "hull": deepcopy(HULL),
    }

    return workspace, cfg


@dataclass(frozen=True)
class SimulationContract:
    """Portable wire contract for engine/UI host migration, not a physics model."""

    version: str = "1.0"
    frame: str = "ENU"
    position_units: str = "m"
    velocity_units: str = "m/s"
    angular_units: str = "rad"

    def serialize(self) -> dict[str, str]:
        return {
            "version": self.version,
            "reference_frame": self.frame,
            "position_units": self.position_units,
            "velocity_units": self.velocity_units,
            "angular_units": self.angular_units,
        }


def build_browser_config(controls: Mapping[str, Any]) -> dict[str, Any]:
    """Generate an independent, serializable engine configuration.

    The physics engine never reads Streamlit widget state. This function is safe
    to reuse from a CLI runner, API server, or native app integration.
    """
    config = deepcopy(dict(controls))
    config["contract"] = SimulationContract().serialize()
    if config["physicsMode"] not in {"baseline", "advanced", "6dof"}:
        raise ValueError("Unsupported BLUEVECTOR dynamics mode")
    if not isinstance(config["hull"], dict) or config["hull"].get("mass", 0) <= 0:
        raise ValueError("Invalid vessel specification")
    return config


# -----------------------------------------------------------------------------
# PORTABLE PRESENTATION ASSETS: named, extractable browser resources.
# BROWSER_PAGE_TEMPLATE = HTML markup; BROWSER_CSS = presentation;
# BROWSER_CONTROLLER_JS = client orchestration. Physics engines remain separate
# UMD libraries above. All are embedded at runtime to retain single-file deploy.
# -----------------------------------------------------------------------------
BROWSER_CSS = r'''
:root{--royal:#4169e1;--bright:#5caaff;--cyan:#5ce8f7;--navy:#06162a;--panel:#0d2540;--line:#294b6a;--text:#f0f7ff;--muted:#a8c1d8;--amber:#ffd479;--red:#ff7f92;--green:#76e6b4}
*{box-sizing:border-box} html,body{padding:0;margin:0;background:var(--navy);font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;color:var(--text)}
button{font:inherit;cursor:pointer}button:focus-visible{outline:3px solid var(--cyan);outline-offset:2px}
button:disabled{opacity:.42;cursor:not-allowed}
.shell{max-width:1470px;margin:auto;padding:10px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.brand{display:flex;align-items:center;gap:10px}
.logo{display:grid;place-items:center;width:42px;height:42px;border-radius:12px;background:linear-gradient(140deg,#477af3,#173c95);font-size:16px;font-weight:850;letter-spacing:-1px;box-shadow:inset 0 0 0 1px #a4d7ff30}
h1{font-size:20px;margin:0;font-weight:750;letter-spacing:.15px} .sub{font-size:11px;color:var(--muted);letter-spacing:1.2px;text-transform:uppercase;margin-top:2px}
.header-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pill{background:#173657;border:1px solid var(--line);padding:6px 9px;border-radius:100px;font-size:11px;color:#d5e9ff}
.statusdot{display:inline-block;width:8px;height:8px;background:var(--green);border-radius:100%;margin-right:5px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 298px;gap:10px;align-items:start}
.main-col{min-width:0;display:flex;flex-direction:column;gap:9px}
.map-frame{position:relative;overflow:hidden;border:1px solid #4576a1;border-radius:15px;background:#0d3555;box-shadow:0 8px 35px #0005}
#chart{display:block;width:100%;height:auto;aspect-ratio:16/9;touch-action:pan-y;cursor:default}
#chart.planning{cursor:crosshair;touch-action:none}
.map-label{position:absolute;top:11px;left:12px;background:#031c32ce;border:1px solid #578bb74e;border-radius:9px;padding:7px 10px;pointer-events:none}
.map-label strong{font-size:11px;letter-spacing:1px;display:block}.map-label span{font-size:10px;color:#a5cbe8}
.map-readout{position:absolute;right:11px;bottom:11px;display:flex;gap:10px;background:#041d33e8;border:1px solid #5c93b87a;padding:8px 11px;border-radius:10px;pointer-events:none;font-variant-numeric:tabular-nums}
.map-readout>div{display:flex;flex-direction:column;gap:1px}.map-readout strong{font-size:14px;color:#fff}.map-readout span{font-size:9px;color:#bad6e9;text-transform:uppercase;letter-spacing:.7px}
.badge{font-size:10px;color:#eef7ff;background:#164163;border:1px solid #37678a;padding:5px 9px;border-radius:6px;letter-spacing:.5px}
.toolbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;background:var(--panel);border:1px solid var(--line);padding:9px;border-radius:12px}
.tool-label{font-size:10px;color:var(--muted);letter-spacing:.9px;margin-right:2px;white-space:nowrap}
.action{background:#173a5d;border:1px solid #42688d;border-radius:9px;color:var(--text);padding:8px 11px;font-weight:650;font-size:12px;min-height:36px;white-space:nowrap}
.action:hover{background:#224d78}.action.selected{background:#234bb3;border-color:#80a8ff;box-shadow:inset 0 0 0 1px #6aa7ff4d}
.action.warn{background:#503037;border-color:#84515a}.action.good{background:#175341;border-color:#3f9677}
.flex-space{flex:1}
.helm{display:flex;justify-content:space-between;gap:12px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 13px;min-height:158px}
.helm-left{display:flex;align-items:center;gap:18px;min-width:0}
.helm-title{font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
.helm-mode{font-size:13px;font-weight:750}.helm-sub{font-size:11px;color:var(--muted);max-width:230px;line-height:1.45}
.joypad{position:relative;width:125px;height:125px;flex:0 0 125px;border-radius:100%;background:radial-gradient(circle,#133a66 0,#102b4b 62%,#0a1b30 100%);border:2px solid #446d9b;box-shadow:inset 0 3px 12px #0008;touch-action:none;user-select:none}
.joypad:after{content:"";position:absolute;left:50%;top:50%;width:2px;height:2px;background:#9bc9f4;transform:translate(-50%,-50%);pointer-events:none}
.joyknob{position:absolute;left:50%;top:50%;width:50px;height:50px;margin:-25px 0 0 -25px;border-radius:100%;background:linear-gradient(145deg,#5e96ff,#274aa9);border:2px solid #a6c7ff;box-shadow:0 3px 12px #0008;display:grid;place-items:center;font-size:20px;pointer-events:none}
.dpad{display:grid;grid-template-columns:repeat(3,39px);grid-template-rows:repeat(3,39px);gap:4px;touch-action:none}
.dpad button{width:39px;height:39px;border:1px solid #6695c8;border-radius:8px;background:#234b86;color:#fff;font-size:19px;touch-action:none}
.dpad button:active{background:#4778c7}
.right-actions{display:flex;align-items:center;flex-wrap:wrap;justify-content:end;gap:7px}
.right-actions .action{min-height:42px}
.aside{display:flex;flex-direction:column;gap:10px;min-width:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px}
.panel-title{font-size:11px;text-transform:uppercase;letter-spacing:1.4px;color:#8fb4d7;margin:0 0 11px;font-weight:750}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.metric{background:#091d34;border:1px solid #244464;border-radius:9px;padding:10px}
.metric strong{font-size:21px;font-variant-numeric:tabular-nums;color:#e5f6ff}.metric small{font-size:10px;color:#aecce3;display:block;margin-top:3px}
.telemetry-row{display:flex;align-items:center;justify-content:space-between;gap:5px;padding:7px 0;border-bottom:1px solid #244464;color:#bed4e8;font-size:11px}
.telemetry-row:last-child{border:none}.telemetry-row b{color:#fff;font-size:12px;font-variant-numeric:tabular-nums}
.health{display:flex;gap:5px;flex-wrap:wrap;margin-top:9px}.health .badge{font-size:10px}
.heading-adjust{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:11px}.heading-adjust button{padding:6px 10px}
#trend{display:block;width:100%;height:94px;border:1px solid #2e5679;border-radius:8px;background:#06192a}
.logs{font-size:11px;color:#c3dbec;min-height:42px;max-height:100px;overflow-y:auto;line-height:1.55}
.note{font-size:10px;color:#95b5cd;line-height:1.45}
.toast{margin:6px 0 0;font-size:11px;color:var(--amber);min-height:15px}
.footer{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:10px;color:#8baecb;margin-top:8px;flex-wrap:wrap}
.footer a{color:#85bbff}
.workspace-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:9px 10px;margin-bottom:9px;background:linear-gradient(100deg,#0b2541,#123c71 55%,#0c2440);border:1px solid #3d6eae;border-radius:13px;box-shadow:0 3px 18px #020b1e70}
.workspace-bar .action{min-height:32px;padding:7px 11px}.workspace-bar .action.selected{background:#4169e1;border-color:#b0d3ff}.speed-wrap{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:11px;color:#c4def2}.speed-wrap select,.analytics-head select{color:#eaf6ff;background:#102b51;border:1px solid #4f7cae;padding:6px;border-radius:8px;font-size:12px}
.analysis-screen,.mission-screen{display:none;background:var(--panel);border:1px solid #355c82;border-radius:12px;padding:12px;margin-top:8px}.shell[data-workspace="analysis"] .analysis-screen,.shell[data-workspace="mission"] .mission-screen{display:block}.shell[data-workspace="analysis"] .views,.shell[data-workspace="analysis"] .helm,.shell[data-workspace="analysis"] .toolbar,.shell[data-workspace="analysis"] .view-bar{display:none}.shell[data-workspace="analysis"] .layout{grid-template-columns:1fr 298px}.analytics-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.analysis-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:12px 0}.analysis-grid .metric{min-width:0}.analysis-grid .metric strong{font-size:18px}.analysis-canvas{width:100%;height:240px;background:#061b2f;border:1px solid #2a5677;border-radius:11px;display:block}.info-inline{color:#a9cbe7;font-size:11px;line-height:1.5}.mission-screen .toolbar{margin-top:9px}.panel button:focus-visible{outline:2px solid #69e8ff}@media(max-width:920px){.shell[data-workspace="analysis"] .layout{grid-template-columns:1fr}.analysis-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.workspace-bar .action{font-size:10px;min-height:31px;padding:6px 8px}.speed-wrap{margin-left:0}.analysis-canvas{height:190px}.analysis-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:920px){.layout{grid-template-columns:1fr}.aside{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));align-items:start}.aside .panel:last-child{grid-column:1/-1}}
@media(max-width:600px){.shell{padding:7px}h1{font-size:15px}.logo{width:34px;height:34px;font-size:18px}.sub{font-size:9px}.header-right{gap:5px}.pill{padding:4px 7px;font-size:9px}.map-label{top:5px;left:5px;padding:5px}.map-label span{display:none}.map-readout{right:5px;bottom:5px;gap:7px;padding:5px 7px}.map-readout strong{font-size:11px}.map-readout span{font-size:8px}.toolbar{gap:5px;padding:7px}.action{padding:7px 8px;font-size:11px;min-height:34px}.helm{padding:9px;gap:7px}.helm-left{gap:7px}.helm-sub{display:none}.joypad{width:112px;height:112px;flex-basis:112px}.right-actions{max-width:185px}.right-actions .action{min-height:36px}.aside{grid-template-columns:1fr;gap:7px}.aside .panel{padding:10px}.aside .panel:nth-child(2){grid-column:auto}.aside .panel:last-child{grid-column:auto}.metric strong{font-size:18px}}

/* BLUEVECTOR mission challenges: shared vessel state, no duplicate physics. */
.challenge-screen{display:none;background:var(--panel);border:1px solid #38668b;border-radius:12px;padding:13px;margin-top:8px}
.shell[data-workspace="challenge"] .challenge-screen{display:block}
.challenge-config{display:flex;align-items:end;flex-wrap:wrap;gap:8px;margin:12px 0}
.challenge-config label{font-size:10px;letter-spacing:.65px;color:#aed5f2;display:flex;flex-direction:column;gap:5px}
.challenge-config select{background:#091d34;border:1px solid #477bab;color:#ecf7ff;border-radius:8px;padding:9px;min-height:36px;min-width:143px;font:inherit;font-size:12px}
.challenge-details{border:1px solid #3c78a3;background:linear-gradient(90deg,#0a2a49,#123d65);border-radius:10px;padding:12px;display:flex;justify-content:space-between;align-items:center;gap:12px}
.challenge-details strong{display:block;font-size:14px;color:#effbff}
.challenge-details span{display:block;font-size:11px;color:#a5cee9;margin-top:5px;line-height:1.5}
.challenge-live{text-align:right;min-width:110px}.challenge-live>div{font-size:10px;color:#72eec4;letter-spacing:1px}.challenge-live strong{font-size:24px}
.challenge-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:9px}
.challenge-result{background:#0c3a3c;border:1px solid #5abfa6;padding:11px;border-radius:10px;margin-top:9px;color:#e9fff7;line-height:1.6;font-size:12px}
.challenge-result[hidden]{display:none}
@media(max-width:650px){.challenge-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.challenge-config label{flex:1 1 130px}.challenge-details{align-items:flex-start}.challenge-live strong{font-size:20px}}

/* BLUEVECTOR 3D bridge: a second view of the SAME simulation state. */
.view-bar{display:flex;align-items:center;flex-wrap:wrap;gap:7px;border:1px solid #345b7f;border-radius:12px;background:linear-gradient(95deg,#102d50,#0b2039);padding:8px 10px}
.view-title{font-size:10px;letter-spacing:1.15px;color:#acd8f6;font-weight:700;margin-right:4px}
.view-buttons{display:flex;gap:5px;flex-wrap:wrap}.view-buttons .action{min-height:31px;padding:6px 10px}
.view-note{margin-left:auto;font-size:10px;color:#a1bed6}
.views{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:stretch}
.views.only-2d,.views.only-3d{grid-template-columns:1fr}
.views.only-2d #twinFrame,.views.only-3d #mapFrame{display:none}
.views .map-frame,.views .twin-frame{min-width:0}
.views:not(.only-2d) .map-frame{display:flex;flex-direction:column;justify-content:center}
.views:not(.only-2d) #chart{aspect-ratio:16/10}
.twin-frame{background:radial-gradient(ellipse at 52% 35%,#14395f,#061529 74%);border:1px solid #4576a1;border-radius:15px;overflow:hidden;position:relative;min-height:510px;display:flex;flex-direction:column}
.twin-header{display:flex;flex-wrap:wrap;align-items:center;gap:7px;justify-content:space-between;padding:10px 11px 5px;background:#051729bb}
.twin-header strong{font-size:11px;letter-spacing:1px;color:#bce7ff}
.twin-header span{font-size:10px;color:#96c0dd}
.camera-bar{display:flex;gap:5px;flex-wrap:wrap;padding:4px 10px 5px;background:#061a2e}
.camera-bar .action{padding:5px 8px;min-height:28px;font-size:10px}
#twinPlot,#threeView,#twinCanvas{height:405px;min-height:405px;flex:1;touch-action:none;position:relative}
#twinCanvas{display:block;width:100%;background:#071c30;cursor:grab}#twinCanvas:active{cursor:grabbing}
#twinPlot[hidden],#threeView[hidden],#twinCanvas[hidden]{display:none!important}#threeView canvas{display:block;width:100%;height:100%;touch-action:none}
#threeView .overlay-label{position:absolute;bottom:10px;left:12px;color:#9ad5ff;font-size:10px;background:#071b32d9;border:1px solid #2773a0;padding:6px 9px;border-radius:7px;pointer-events:none}
#twinPlot .js-plotly-plot,#twinPlot .plot-container{height:100%!important}
.twin-status{background:#092b47;color:#bce9f5;font-size:10px;letter-spacing:.35px;border-top:1px solid #1c5681;padding:5px 12px;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}.twin-status button{background:#143c65;color:#daf5ff;border:1px solid #5b8eb2;border-radius:6px;padding:4px 8px;font-size:10px}.twin-message{position:absolute;left:10px;top:86px;right:10px;background:#061e36e6;border:1px solid #457da9;border-radius:9px;padding:11px;color:#bfe8ff;text-align:center;font-size:12px;line-height:1.5;z-index:2;pointer-events:none}
.twin-info{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;padding:7px 11px;border-top:1px solid #264964;background:#071c32;font-size:10px;color:#b9d7eb}
.twin-info strong{color:#e3f7ff}
.layer-picks{display:flex;flex-wrap:wrap;gap:4px 11px;padding:7px 10px 10px;background:#081d36;border-top:1px solid #203e5c}
.layer-picks label{font-size:10px;color:#b9d5ec;display:flex;align-items:center;gap:4px;white-space:nowrap;cursor:pointer;user-select:none}
.layer-picks input{accent-color:#5ce8f7;width:13px;height:13px;margin:0}
.plotly .modebar{top:4px!important;right:6px!important}
@media(max-width:1220px){.views{grid-template-columns:1fr}.twin-frame{min-height:420px}#twinPlot,#threeView,#twinCanvas{height:340px;min-height:340px}.views:not(.only-3d) #chart{aspect-ratio:16/9}}
@media(max-width:600px){.view-bar{padding:7px;gap:5px}.view-note{width:100%;margin:0}.view-buttons .action{min-height:29px;padding:5px 8px}.twin-frame{min-height:395px}#twinPlot,#threeView,#twinCanvas{height:310px;min-height:310px}.layer-picks{gap:7px 10px}.twin-header{gap:4px}}


/* Linked Plotly compass: same ENU true heading and user camera as the USV. */
.magnetic-lab{border-top:1px solid #305c82;background:#081a2e;padding:8px 10px}
.magnetic-lab-title{display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap;color:#ccedfc;font-size:10px;letter-spacing:.5px}
.magnetic-lab-title label{font-size:10px;color:#a7dcf2;display:flex;align-items:center;gap:5px;cursor:pointer}
.magnetic-lab input{accent-color:#52d6fd}
.magnetic-lab-sliders{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0 4px}
.magnetic-lab-sliders label{font-size:10px;color:#a6cfe5;display:flex;flex-direction:column;gap:4px;min-width:0}
.magnetic-lab-sliders label span{color:#80efb2}
.magnetic-lab-sliders input[type=range]{width:100%;min-width:0}
.magnetic-lab .compass-note{background:transparent;padding:4px 0;font-size:9.5px}
.allocation-priority{font-size:10px;color:#a6cfe5;display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:4px 0}
.allocation-priority select{font:inherit;background:#12385b;border:1px solid #386e99;color:#e2f3ff;border-radius:6px;padding:5px 7px}
@media(max-width:600px){.magnetic-lab-sliders{grid-template-columns:1fr 1fr}}
/* Combined USV plus true-north compass is the primary visualization. */
.twin-pair{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;min-width:0}
.twin-pair .compass-frame[hidden]{display:none!important}
.twin-pair:has(.compass-frame:not([hidden])){grid-template-columns:minmax(0,1.48fr) minmax(300px,.92fr)}
.combined-compass-strip{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:8px 11px;border-top:1px solid #346b95;background:linear-gradient(90deg,#0b2948,#081a30);font-size:10px;color:#9fd3ed}
.combined-compass-strip strong{letter-spacing:.3px;color:#e0f5ff}
.combined-compass-strip .legend{white-space:nowrap}
.combined-compass-strip .legend-cyan{color:#48eafc}
.combined-compass-strip .legend-green{color:#8ef7b6}
.combined-compass-strip .legend-amber{color:#ffdc87}
.combined-compass-strip button{margin-left:auto;background:#184575;color:#e5f6ff;border:1px solid #5e98c1;border-radius:7px;padding:6px 11px;font-size:10px;cursor:pointer;min-height:28px}
.combined-compass-strip button:hover{background:#236aa4}

.twin-pair .twin-frame{min-width:0}
.views:not(.only-3d) .twin-pair{grid-template-columns:1fr}
.views.only-2d .twin-pair{display:none}
.compass-frame{border-color:#327cc1;background:radial-gradient(ellipse at 50% 35%,#173f70,#06172a 73%)}
#compassPlot,#compassFallback{width:100%;height:405px;min-height:405px;flex:1;touch-action:none;background:#071c30}
#compassFallback{display:block}
#compassPlot[hidden],#compassFallback[hidden]{display:none!important}
#compassPlot .js-plotly-plot,#compassPlot .plot-container{height:100%!important}
.compass-command{display:grid;grid-template-columns:1fr auto;align-items:center;gap:9px;background:#0a213b;padding:10px 12px;border-top:1px solid #244e78}
.compass-command .dial-label{font-size:10px;color:#afd9f4;letter-spacing:.45px}
.compass-command input[type=range]{accent-color:#4ecaff;width:100%;cursor:pointer;touch-action:pan-x}
.compass-command .cmd-value{font-size:15px;font-weight:800;color:#ffda8c;min-width:50px;text-align:right}
.compass-status{font-size:10px;color:#b6dff9;background:#06233c;padding:7px 11px;border-top:1px solid #2e679a;line-height:1.5}
.compass-status strong{color:#70f7cd}
.compass-tools{display:flex;align-items:center;gap:5px;flex-wrap:wrap;justify-content:flex-start}
.compass-tools .action{padding:5px 9px;font-size:10px;min-height:27px}
.compass-tools .action.selected{background:#4169e1}
.compass-note{font-size:10px;color:#97c5df;padding:7px 11px;background:#06172b;line-height:1.5}
@media(max-width:1270px){.twin-pair:has(.compass-frame:not([hidden])){grid-template-columns:1fr 1fr}#compassPlot,#compassFallback{height:340px;min-height:340px}}
@media(max-width:950px){.twin-pair:has(.compass-frame:not([hidden])){grid-template-columns:1fr}#compassPlot,#compassFallback{height:340px;min-height:340px}}
@media(max-width:600px){#compassPlot,#compassFallback{height:300px;min-height:300px}.compass-frame{min-height:420px}}

'''

LINEAR_ALGEBRA_ENGINE = r'''
/* BLUEVECTOR Matrix Laboratory: portable linear-algebra primitives and diagnostics.
 * Body axes x forward, y port, z up; world is East/North/Up.
 * Heading clockwise from true north. All internal angles are radians.
 * Explicitly a demonstration sensor/covariance model, not a certified INS.
 */
(function(root,factory){const api=factory();if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.BluevectorLinear=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const PI=Math.PI,G=9.80665;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
const bearing=x=>((x%360)+360)%360;
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const transpose=A=>A[0].map((_,j)=>A.map(row=>row[j]));
const matmul=(A,B)=>A.map(row=>B[0].map((_,j)=>row.reduce((s,x,k)=>s+x*B[k][j],0)));
const matvec=(A,v)=>A.map(row=>dot(row,v));
const zeros=(n,m)=>Array.from({length:n},()=>Array(m).fill(0));
const identity=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
function rotation(q){
 if(!Array.isArray(q)||q.length!==4)throw Error('quaternion must be [w,x,y,z]');
 const a=Math.hypot(...q);if(!(a>1e-12))throw Error('zero quaternion');
 const [w,x,y,z]=q.map(v=>v/a);
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}
function fromEuler(roll,pitch,yaw){
 const a=Math.cos(roll/2),b=Math.sin(roll/2),c=Math.cos(pitch/2),d=Math.sin(pitch/2),e=Math.cos(yaw/2),f=Math.sin(yaw/2);
 return [a*c*e+b*d*f,b*c*e-a*d*f,a*d*e+b*c*f,a*c*f-b*d*e];
}
function transform(q,p,translation=[0,0,0]){
 const r=matvec(rotation(q),p);return r.map((x,i)=>x+translation[i]);
}
function headingFromQuaternion(q){const forward=matvec(rotation(q),[1,0,0]);const flat=Math.hypot(forward[0],forward[1]);
 if(flat<1e-9)return null;return bearing(Math.atan2(forward[0],forward[1])*180/PI);
}
function determinant3(M){return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);}
function invert3(M){const det=determinant3(M);if(Math.abs(det)<1e-10)throw Error('singular calibration matrix');
 const out=zeros(3,3);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++){
  const rows=[0,1,2].filter(r=>r!==j),cols=[0,1,2].filter(c=>c!==i);
  out[i][j]=((i+j)%2?-1:1)*(M[rows[0]][cols[0]]*M[rows[1]][cols[1]]-M[rows[0]][cols[1]]*M[rows[1]][cols[0]])/det;
 }return out;
}
/* Complete 2x2 bounded, weighted least-squares control allocation.
 * Desired surge force and yaw moment in body axes. B=[[1,1],[-a,a]].
 * Normalize force and moment to avoid dimensional weighting errors.
 * For 2 motors, optimum is either interior or on one of the four box edges.
 */
function allocate(surgeN,yawNm,arm,maxThrust,options={}){
 if(!(arm>0&&maxThrust>0))throw Error('arm/maxThrust must be positive');
 const limit=clamp(Number(options.limitFraction??1),0,1)*maxThrust;
 const wx=Math.max(0.01,Number(options.surgeWeight??1)),wn=Math.max(0.01,Number(options.yawWeight??1)),
       lambda=Math.max(0,Number(options.regularization??1e-11));
 const fscale=2*maxThrust,mscale=2*arm*maxThrust;
 const v=[surgeN/fscale,yawNm/mscale];
 const A=[[wx/fscale,wx/fscale],[-wn*arm/mscale,wn*arm/mscale]],
       b=[wx*v[0],wn*v[1]];
 const H=[[dot([A[0][0],A[1][0]],[A[0][0],A[1][0]])+lambda,
           dot([A[0][0],A[1][0]],[A[0][1],A[1][1]])],
          [0,dot([A[0][1],A[1][1]],[A[0][1],A[1][1]])+lambda]];
 H[1][0]=H[0][1];
 const g=[A[0][0]*b[0]+A[1][0]*b[1],A[0][1]*b[0]+A[1][1]*b[1]];
 const obj=(p,s)=>{const e0=(p+s)/fscale-v[0],e1=arm*(s-p)/mscale-v[1];return wx*wx*e0*e0+wn*wn*e1*e1+lambda*(p*p+s*s);};
 const choices=[];
 const det=H[0][0]*H[1][1]-H[0][1]**2;
 if(det>1e-15){const p=(H[1][1]*g[0]-H[0][1]*g[1])/det,s=(H[0][0]*g[1]-H[0][1]*g[0])/det;
  if(Math.abs(p)<=limit&&Math.abs(s)<=limit)choices.push([p,s]);}
 for(const f of [-limit,limit]){
  choices.push([f,clamp((g[1]-H[1][0]*f)/H[1][1],-limit,limit)]);
  choices.push([clamp((g[0]-H[0][1]*f)/H[0][0],-limit,limit),f]);
 }
 if(limit===0)choices.push([0,0]);
 choices.sort((a,b)=>obj(a[0],a[1])-obj(b[0],b[1]));
 const [port,starboard]=choices[0];
 return {port,starboard,requestedSurge_N:surgeN,requestedYaw_Nm:yawNm,
   deliveredSurge_N:port+starboard,deliveredYaw_Nm:arm*(starboard-port),
   forceError_N:port+starboard-surgeN,momentError_Nm:arm*(starboard-port)-yawNm,
   saturation:Math.max(Math.abs(port),Math.abs(starboard))>=limit-1e-6};
}
function magneticParams(hardIron_uT=0,softIron_percent=0){
 const h=clamp(hardIron_uT,0,25),d=clamp(softIron_percent,0,35)/100;
 return {A:[[1+d,.35*d,.04*d],[.35*d,1-.6*d,.08*d],[.04*d,.08*d,1+.2*d]],bias:[h,-.6*h,.2*h]};
}
function earthField(declinationDegrees=0,inclinationDegrees=55,intensity_uT=50){
 const d=declinationDegrees*PI/180,i=inclinationDegrees*PI/180;
 return [intensity_uT*Math.cos(i)*Math.sin(d),intensity_uT*Math.cos(i)*Math.cos(d),-intensity_uT*Math.sin(i)];
}
function simulateMagnetometer(q,settings={},noise=[0,0,0]){
 const field=earthField(settings.declination_deg??0,settings.inclination_deg??55,settings.intensity_uT??50),
       R=rotation(q),body=matvec(transpose(R),field),p=magneticParams(settings.hardIron_uT??0,settings.softIron_percent??0);
 const raw=matvec(p.A,body).map((v,i)=>v+p.bias[i]+noise[i]);
 const corrected=settings.calibrated===false?raw:matvec(invert3(p.A),raw.map((v,i)=>v-p.bias[i]));
 const {roll=0,pitch=0}=settings;
 const tilt=rotation(fromEuler(roll,pitch,0));
 const level=matvec(tilt,corrected);
 const magBearing=Math.atan2(level[1],level[0])*180/PI;
 const trueHeading=bearing(magBearing+(settings.declination_deg??0));
 return {raw,calibratedField:corrected,estimatedTrueHeading_deg:trueHeading,
         fieldNorm_uT:Math.hypot(...corrected),distortionNorm_uT:Math.hypot(...raw)-Math.hypot(...body),
         calibrated:settings.calibrated!==false};
}
/* 15-state diagnostic ESKF covariance ONLY. The existing 4-state EKF remains
 * the authoritative navigation position and heading estimate. Covariance
 * order: position ENU(3), velocity ENU(3), attitude body(3), gyro bias(3),
 * accelerometer bias(3). No claim to a full integrated GNSS/INS solution.
 */
const PSDIM=15;
function initCovariance(){const diag=[4,4,9,.4,.4,.4,.03**2,.03**2,(3*PI/180)**2,
  .003**2,.003**2,.003**2,.05**2,.05**2,.05**2];
 return diag.map((v,i)=>Array.from({length:PSDIM},(_,j)=>i===j?v:0));}
function covariancePredict(P,dt,q=[1,0,0,0],omega=[0,0,0]){
 if(P.length!==PSDIM||P.some(row=>row.length!==PSDIM))throw Error('15x15 covariance required');
 const F=identity(PSDIM),R=rotation(q);
 for(let k=0;k<3;k++){
  F[k][3+k]=dt;
  for(let j=0;j<3;j++)F[3+k][12+j]=-R[k][j]*dt;
  F[6+k][9+k]=-dt;
 }
 const S=[[0,-omega[2],omega[1]],[omega[2],0,-omega[0]],[-omega[1],omega[0],0]];
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)F[6+i][6+j]-=dt*S[i][j];
 const out=matmul(matmul(F,P),transpose(F));
 const qdiag=[0,0,0,.12**2,.12**2,.12**2,(.007)**2,(.007)**2,(.007)**2,
              (.00025)**2,(.00025)**2,(.00025)**2,(.002)**2,(.002)**2,(.002)**2];
 for(let i=0;i<PSDIM;i++)out[i][i]+=qdiag[i]*dt;
 return symmetrize(out);
}
function symmetrize(P){return P.map((row,i)=>row.map((v,j)=>(v+P[j][i])/2));}
function covarianceScalarUpdate(P,index,noiseVariance){
 if(index<0||index>=PSDIM||!(noiseVariance>0))throw Error('invalid covariance measurement');
 const S=P[index][index]+noiseVariance,K=P.map(row=>row[index]/S),
       A=identity(PSDIM);
 for(let i=0;i<PSDIM;i++)A[i][index]-=K[i];
 const out=matmul(matmul(A,P),transpose(A));
 for(let i=0;i<PSDIM;i++)for(let j=0;j<PSDIM;j++)out[i][j]+=K[i]*noiseVariance*K[j];
 return symmetrize(out);
}
function covarianceMetrics(P){return {position_sigma_m:Math.sqrt(Math.max(0,(P[0][0]+P[1][1])/2)),
 heading_sigma_deg:Math.sqrt(Math.max(0,P[8][8]))*180/PI,
 tilt_sigma_deg:Math.sqrt(Math.max(0,(P[6][6]+P[7][7])/2))*180/PI,
 covariance_trace:P.reduce((acc,row,i)=>acc+row[i],0)};}
return Object.freeze({rotation,fromEuler,transform,headingFromQuaternion,determinant3,invert3,transpose,matmul,matvec,allocate,
 magneticParams,earthField,simulateMagnetometer,initCovariance,covariancePredict,covarianceScalarUpdate,covarianceMetrics});
});
'''


BROWSER_CONTROLLER_JS = r'''
(() => {
'use strict';
const cfg = JSON.parse(document.getElementById('cfg').textContent);
let simRate=1; let activeWorkspace="twin"; let lastAnalysisDraw=0;
const $ = id => document.getElementById(id);
// WebAudio is user-gesture gated. Android browsers may support vibration;
// iOS Safari normally does not expose navigator.vibrate.
const AudioEngine=window.AudioContext||window.webkitAudioContext;
let audioCtx=null,muted=false;
function unlockAudio(){
 if(muted||!AudioEngine)return;
 try{if(!audioCtx)audioCtx=new AudioEngine();if(audioCtx.state==='suspended')audioCtx.resume().catch(()=>{});}catch(e){}
}
function chirp(hz=670,duration=.09,kind='sine',volume=.024){
 if(muted||!audioCtx||audioCtx.state!=='running')return;
 try{
  const osc=audioCtx.createOscillator(),gain=audioCtx.createGain(),t=audioCtx.currentTime;
  osc.type=kind;osc.frequency.setValueAtTime(hz,t);
  gain.gain.setValueAtTime(0,t);gain.gain.linearRampToValueAtTime(volume,t+.012);
  gain.gain.exponentialRampToValueAtTime(.0001,t+duration);
  osc.connect(gain);gain.connect(audioCtx.destination);osc.start(t);osc.stop(t+duration+.015);
 }catch(e){}
}
function haptic(pattern=20){if(navigator.vibrate)try{navigator.vibrate(pattern);}catch(e){}}
function feedback(type){
 if(type==='waypoint'){chirp(740,.09);haptic(18);}
 else if(type==='complete'){chirp(950,.20,'triangle',.04);haptic([70,65,70]);}
 else if(type==='failsafe'){chirp(260,.28,'sawtooth',.018);haptic([70,50,70]);}
 else if(type==='lamp'){chirp(630,.07,'triangle');haptic(10);}
 else if(type==='mode'){chirp(480,.06);haptic(10);}
}
window.addEventListener('pointerdown',unlockAudio,{capture:true});
window.addEventListener('keydown',unlockAudio,{capture:true});
const canvas = $('chart'), ctx = canvas.getContext('2d');
const trend = $('trend'), tctx = trend.getContext('2d');
const analysisCanvas=$('analysisCanvas'),actx=analysisCanvas.getContext('2d');
const CW=1200, CH=675, SCALE=1200/180, WORLD={w:180,h:675/SCALE};
const DT=1/60, TAU=Math.PI*2;
const LA=window.BluevectorLinear; // portable matrix, allocator and sensor library
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const rad=d=>d*Math.PI/180, deg=r=>r*180/Math.PI;
const heading=psi=>(90-deg(psi)+720)%360;
const psiOfHeading=h=>rad(90-h);
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const toX=x=>x*SCALE, toY=y=>(WORLD.h-y)*SCALE;
const vec=(x,y)=>({x,y});
const nfmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—';
let randomState=cfg.seed>>>0;
function rng(){randomState=(randomState+0x6D2B79F5)>>>0;let t=randomState;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
function normal(){return Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(TAU*rng());}
// Positive yaw means turn to port, with East-North-Up chart conventions.
// Effective diagonal inertia and drag coefficients are illustrative, not measured.
// Single parameter source is also used by the Python research simulator.
const hull={...cfg.hull,maxThrust:cfg.maxThrust,motorTau:cfg.motorTau};
const current={x:cfg.currentE,y:cfg.currentN};
// Magnetic sensor and navigation covariance are diagnostics, distinct from true heading.
const magSettings={hardIron_uT:0,softIron_percent:0,declination_deg:0,calibrated:true};
let allocationPriority="balanced";
let magneticSample={estimatedTrueHeading_deg:90,raw:[0,0,0],fieldNorm_uT:50,
 distortionNorm_uT:0,calibrated:true,seq:0};
let allocatedControl=null;
function sampleMagneticHeading(){
 const b=s.boat,q=b.attitude||LA.fromEuler(b.roll||0,b.pitch||0,b.psi);
 const noise=Array.from({length:3},()=>normal()*.09);
 const sample=LA.simulateMagnetometer(q,{...magSettings,roll:b.roll||0,pitch:b.pitch||0},noise);
 magneticSample={...sample,seq:magneticSample.seq+1};
 return psiOfHeading(sample.estimatedTrueHeading_deg);
}
let waveAmplitude=cfg.waveAmp;
function currentAt(x,y,t){
 if(!cfg.currentField)return {x:current.x,y:current.y};
 return {x:current.x+0.11*Math.sin(y/18+t/85),y:current.y+0.075*Math.cos(x/24-t/90)};
}
// Local spatial/time gradients of the SAME current field used by kinematics.
// Required for the rigid-body inertia term in a current-relative six-axis model.
function currentDerivativesAt(x,y,t){
 if(!cfg.currentField)return {dxE:0,dyE:0,dtE:0,dxN:0,dyN:0,dtN:0};
 const a=y/18+t/85,b=x/24-t/90;
 return {dxE:0,dyE:.11*Math.cos(a)/18,dtE:.11*Math.cos(a)/85,
   dxN:-.075*Math.sin(b)/24,dyN:0,dtN:.075*Math.sin(b)/90};
}
const windTo=rad(cfg.windFrom+180);
const wind={x:cfg.windSpeed*Math.sin(windTo),y:cfg.windSpeed*Math.cos(windTo)};
const obstacles=[{x:68,y:48,r:6.3,name:'Shallow reef'},{x:111,y:42,r:7.5,name:'Restricted zone'},{x:146,y:70,r:5.3,name:'Rocky shoal'}];
const demoWaypoints=()=>[{x:49,y:73},{x:94,y:70},{x:130,y:83},{x:166,y:55}];
const initialVessel=()=>({x:21,y:48,z:0,psi:0,roll:0,pitch:0,u:0,v:0,w:0,p:0,q:0,r:0,energy:0,dist:0,port:0,star:0,power:25,
 motorPort:0,motorStar:0,motorTemp:20,dcVoltage:25.2,rpmPort:0,rpmStar:0,
 usefulPower:0,shaftPower:0,batteryLoss:0,batteryDischarge:25, powertrainScale:1,limitEvents:0});
let controls={throttle:0,steer:0}, keys=Object.create(null);
let joy={x:0,y:0,active:false}, helmMode=cfg.control;
let planning=false, paused=false, lamp=false, lampHeld=false, lampToggle=false, eventLog=[];
let challenge=null;
function initialState(){
 const boat=initialVessel();
 if(cfg.physicsMode==='6dof')window.Bluevector6DOF.initialize(boat);
 return {t:0,boat,est:{x:boat.x,y:boat.y,psi:boat.psi,sigma:cfg.gnssSigma,lastFix:0,gyroBias:0,compassClock:0,gnssClock:0,
  P:[[2,0,0,0],[0,2,0,0],[0,0,rad(3)**2,0],[0,0,0,rad(.25)**2]],
  insCov:LA.initCovariance(),accepted:0,rejected:0,nis:null,filter:cfg.navFilter},
  mode:'manual',targetHeading:90,waypoints:demoWaypoints(),wpIndex:0,navRoute:[],navIndex:0,segmentStart:{x:boat.x,y:boat.y},
  failsafe:null,gnss:true,c2:true,lossTime:0,finished:false,collisions:0,track:[],samples:[],sampleClock:0,traceClock:0,
  risk:'—',xte:null,wpDist:null,desiredSpeed:0,alerts:[],lamp:false,maxXte:0,navErrorSum:0,navErrorN:0,previewRoute:null,
  traffic:cfg.traffic?{x:96,y:24,vx:0.0,vy:0.65,r:2.4}:null};
}
let s=initialState();
function logEvent(msg){
 eventLog.unshift('T+'+nfmt(s.t,1)+'s  '+msg);eventLog=eventLog.slice(0,7);
 $('logs').textContent=eventLog.join('\n');$('logs').style.whiteSpace='pre-line';
 if(msg.startsWith('Waypoint'))feedback('waypoint');
 if(msg==='MISSION COMPLETE')feedback('complete');
 if(msg.startsWith('FAILSAFE'))feedback('failsafe');
 if(msg.startsWith('OBJECTIVE')||msg.startsWith('CHALLENGE'))feedback('waypoint');
}
function notice(msg){$('toast').textContent=msg;}
function setFailsafe(reason){if(s.failsafe)return;s.failsafe=reason;logEvent('FAILSAFE: '+reason);notice('Failsafe idle: '+reason+'. Reconnect and press RESUME.');}
function toggleMode(mode){
 if(challenge&&challenge.status==='running'){notice('Challenge uses locked helm authority. Retry with a different pilot choice.');return;}
 if(mode==='auto'&&!s.waypoints.length){notice('Add at least one waypoint to activate autonomous navigation.');return;}
 if(s.finished&&paused)paused=false;
 s.mode=mode;s.finished=false;s.previewRoute=null;
 if(mode==='heading')s.targetHeading=heading(s.est.psi);
 if(mode==='auto'){s.wpIndex=0;s.navIndex=0;s.segmentStart={x:s.est.x,y:s.est.y};s.navRoute=planRoute(s.segmentStart,s.waypoints);}
 planning=false;refreshButtons();feedback('mode');logEvent(mode==='manual'?'Manual helm enabled':mode==='heading'?'Heading hold enabled':'Waypoint autopilot enabled');
}
function reset(){randomState=cfg.seed>>>0;current.x=cfg.currentE;current.y=cfg.currentN;wind.x=cfg.windSpeed*Math.sin(windTo);wind.y=cfg.windSpeed*Math.cos(windTo);waveAmplitude=cfg.waveAmp;s=initialState();magneticSample={...magneticSample,seq:0};allocatedControl=null;eventLog=[];lamp=false;lampHeld=false;lampToggle=false;challenge=null;paused=false;planning=false;keys=Object.create(null);joy={x:0,y:0,active:false};controls={throttle:0,steer:0};refreshButtons();logEvent('Scenario reset');notice('Reset to demo scenario.');if($('challengeState'))updateChallengeDisplay();}
// ---- USV MISSION CHALLENGES: all conditions use the authoritative vessel state. ----
const challengeBriefs={
 channel:'Pass channel gates in the correct direction without touching markers.',
 inspection:'Maintain stand-off range, illuminate targets, and stabilize your observation.',
 station:'Hold position and heading against current and wind for the required dwell.',
 resilience:'Complete the navigation route through GNSS and C2 disruptions, with supervised recovery.'
};
const CHALLENGE_GATE_COORDS=[{x:48,y:73},{x:93,y:70},{x:129,y:82},{x:150,y:90}];
const CHALLENGE_STATIONS=[{x:49,y:73},{x:93,y:70},{x:129,y:83}];
function buildGates(level){
 const width={basic:17,standard:13,advanced:10}[level];
 return CHALLENGE_GATE_COORDS.map((p,i)=>{
  const from=i?CHALLENGE_GATE_COORDS[i-1]:{x:21,y:48};
  const d=Math.max(.01,dist(from,p)),ux=(p.x-from.x)/d,uy=(p.y-from.y)/d;
  return {...p,ux,uy,width,port:{x:p.x-uy*width/2,y:p.y+ux*width/2},star:{x:p.x+uy*width/2,y:p.y-ux*width/2}};
 });
}
function makeChallenge(type,level,pilot){
 const gates=buildGates(level);
 const targetPositions=CHALLENGE_STATIONS.map((p,i)=>({x:p.x+12,y:p.y,station:{...p},name:'AtoN inspection '+(i+1)}));
 const hard={basic:0,standard:1,advanced:2}[level];
 return {type,level,pilot,status:'running',start:0,elapsed:0,index:0,completed:0,objectives:type==='station'?1:type==='inspection'?3:4,
  gates,targets:targetPositions,station:{x:50,y:55,heading:90},dwell:0,
  dwellRequired:type==='station'?[7,11,15][hard]:[2.5,3.5,5][hard],
  radius:[9,7.5,6][hard],headingTol:rad([48,36,27][hard]),
  safetyWarnings:0,accuracySum:0,accuracyN:0,energyStart:0,distanceStart:0,observations:[],
  recovered:false,gnssTriggered:false,gnssRestore:0,c2Triggered:false,c2Restore:0,lossObserved:false,
  metrics:{completion:0,safety:100,tracking:0,energy:100,timing:100,total:0},score:0,outcome:'',
  prior:{x:21,y:48},lastLog:'',startMode:'manual'};
}
function challengeGoal(ch){
 if(ch.type==='channel')return ch.gates[Math.min(ch.index,ch.gates.length-1)];
 if(ch.type==='inspection')return ch.targets[Math.min(ch.index,ch.targets.length-1)].station;
 if(ch.type==='station')return ch.station;
 return s.waypoints[Math.min(ch.index,s.waypoints.length-1)]||{x:21,y:48};
}
function challengeCommand(ch){
 const goal=challengeGoal(ch),dx=goal.x-s.est.x,dy=goal.y-s.est.y,d=Math.hypot(dx,dy);
 let psi=Math.atan2(dy,dx),speed=clamp(.38*d,0,1.9);
 if(ch.type==='station'&&d<ch.radius*.55){psi=psiOfHeading(ch.station.heading);speed=0;}
 if(ch.type==='inspection'&&d<3.4){const target=ch.targets[ch.index];psi=Math.atan2(target.y-s.est.y,target.x-s.est.x);speed=0;}
 if(d>4){const safe=avoidance(psi,speed);psi=safe.psi;speed=safe.speed;}
 return headingController(psi,speed);
}
function startChallenge(type,level,pilot){
 reset();challenge=makeChallenge(type,level,pilot);
 const ch=challenge;ch.start=s.t;ch.energyStart=s.boat.energy;ch.distanceStart=s.boat.dist;
 if(type==='channel'){
  // The autonomous targets are 5 m beyond each gate, so stopping at a waypoint
  // cannot be mistaken for having passed through the gate itself.
  s.waypoints=ch.gates.map(g=>({x:clamp(g.x+5*g.ux,4,WORLD.w-4),y:clamp(g.y+5*g.uy,4,WORLD.h-4)}));
 }else if(type==='resilience')s.waypoints=demoWaypoints();
 else s.waypoints=[];
 if(pilot==='auto'){
  s.mode='auto';s.wpIndex=0;s.navIndex=0;s.segmentStart={x:s.est.x,y:s.est.y};
  s.navRoute=(type==='channel'||type==='resilience')?planRoute(s.segmentStart,s.waypoints):[];
 }
 ch.startMode=s.mode;
 planning=false;paused=false;refreshButtons();
 logEvent('CHALLENGE START: '+type+' / '+level+' / '+pilot);
 if(type==='inspection')notice('INSPECTION: approach marker at safe stand-off, point vessel and switch searchlight on for the observation dwell.');
 else if(type==='resilience')notice('RESILIENCE: an automatic GNSS interruption and C2 outage will occur. Restore C2, then press RESUME to recover.');
 else notice(challengeBriefs[type]);
 updateChallengeDisplay();
}
function challengeMetric(ch){
 const done=ch.completed,ratio=done/Math.max(1,ch.objectives);
 const completion=clamp(ratio*100,0,100);
 const safety=clamp(100-20*ch.safetyWarnings-100*s.collisions,0,100);
 const error=ch.accuracySum/Math.max(1,ch.accuracyN);
 const tracking=clamp(100-9*error,0,100);
 // An illustrative energy budget normalizes energy per meter, not energy alone.
 const meters=Math.max(1,s.boat.dist-ch.distanceStart);
 const wh=Math.max(0,s.boat.energy-ch.energyStart);
 const targetWhPerM=.045;
 const energy=clamp(100*targetWhPerM/Math.max(targetWhPerM,wh/meters),0,100);
 const targetSecs={channel:125,inspection:180,station:45,resilience:160}[ch.type];
 const timing=clamp(100*targetSecs/Math.max(targetSecs,ch.elapsed),0,100);
 let total=Math.round(.35*completion+.25*safety+.15*tracking+.15*energy+.10*timing);
 if(s.collisions)total=Math.min(total,49);
 return {completion,safety,tracking,energy,timing,total,meanError:error,energyWh:wh};
}
function endChallenge(ch,success,reason){
 if(ch.status!=='running')return;
 ch.status=success?'completed':'failed';ch.outcome=reason;ch.metrics=challengeMetric(ch);ch.score=ch.metrics.total;
 paused=true;refreshButtons();logEvent(success?'CHALLENGE COMPLETE':'CHALLENGE FAILED');
 notice((success?'MISSION ACCOMPLISHED':'MISSION UNSUCCESSFUL')+' · '+reason+' · Score '+ch.score+'/100');
 updateChallengeDisplay();feedback(success?'complete':'failsafe');
}
function updateChallenge(dt,previous){
 const ch=challenge;if(!ch||ch.status!=='running')return;
 ch.elapsed=s.t-ch.start;const b=s.boat;
 if(s.collisions>0){endChallenge(ch,false,'Vessel contact or area-boundary event');return;}
 if(ch.type==='resilience'){
  // Deterministic timed faults. A C2 interruption must not be auto-cleared:
  // command authority is explicitly restored by the operator using RESUME.
  if(!ch.gnssTriggered&&ch.elapsed>=18){ch.gnssTriggered=true;ch.gnssRestore=ch.elapsed+6;s.gnss=false;s.est.gnssClock=0;logEvent('GNSS training outage initiated');refreshButtons();}
  if(ch.gnssTriggered&&!s.gnss&&ch.elapsed>=ch.gnssRestore){s.gnss=true;logEvent('GNSS training outage ended');refreshButtons();}
  if(!ch.c2Triggered&&ch.elapsed>=42){ch.c2Triggered=true;ch.c2Restore=ch.elapsed+3.5;s.c2=false;logEvent('C2 training outage initiated');refreshButtons();}
  if(ch.c2Triggered&&!s.c2&&ch.elapsed>=ch.c2Restore){s.c2=true;logEvent('C2 link restored: press RESUME');notice('C2 RESTORED: press RESUME to acknowledge failsafe and continue.');refreshButtons();}
  if(s.failsafe==='C2 link lost')ch.lossObserved=true;
  if(ch.lossObserved&&!s.failsafe&&s.c2)ch.recovered=true;
 }
 if(ch.type==='channel'){
  const g=ch.gates[ch.index];if(g){
   const before=(previous.x-g.x)*g.ux+(previous.y-g.y)*g.uy;
   const after=(b.x-g.x)*g.ux+(b.y-g.y)*g.uy;
   if(before<0&&after>=0){
    const fraction=-before/Math.max(1e-9,after-before);
    const hit={x:previous.x+(b.x-previous.x)*fraction,y:previous.y+(b.y-previous.y)*fraction};
    const side=Math.abs((hit.x-g.x)*(-g.uy)+(hit.y-g.y)*g.ux);
    // Include half the hull beam and a geometric buffer around the channel posts.
    if(side+hull.beam/2+0.4<=g.width/2){ch.index++;ch.completed++;logEvent('OBJECTIVE Gate '+ch.completed+' passed');}
    else{ch.safetyWarnings++;logEvent('Gate passage outside marked channel');}
   }
  }
  if(ch.completed===ch.objectives){endChallenge(ch,true,'All channel gates passed without contact');return;}
 }else if(ch.type==='inspection'){
  const target=ch.targets[ch.index];if(target){
   const d=dist(b,target),speed=Math.hypot(groundVelocity(b).x,groundVelocity(b).y);
   const bearing=Math.atan2(target.y-b.y,target.x-b.x);
   const correct=lamp&&d>=5&&d<=[16,15,13][['basic','standard','advanced'].indexOf(ch.level)]&&speed<=1.25&&Math.abs(wrap(b.psi-bearing))<rad(55);
   ch.dwell=correct?ch.dwell+dt:Math.max(0,ch.dwell-2*dt);
   if(ch.dwell>=ch.dwellRequired){ch.observations.push({target:target.name,t:s.t,range_m:d,speed_mps:speed});ch.completed++;ch.index++;ch.dwell=0;logEvent('OBJECTIVE Inspection '+ch.completed+' complete');}
  }
  if(ch.completed===ch.objectives){endChallenge(ch,true,'All illuminated inspections verified');return;}
 }else if(ch.type==='station'){
  const d=dist(b,ch.station),speed=Math.hypot(groundVelocity(b).x,groundVelocity(b).y);
  const angle=Math.abs(wrap(b.psi-psiOfHeading(ch.station.heading)));
  const good=d<=ch.radius&&angle<=ch.headingTol&&speed<.9;
  ch.dwell=good?ch.dwell+dt:Math.max(0,ch.dwell-2*dt);
  if(ch.dwell>=ch.dwellRequired){ch.completed=1;endChallenge(ch,true,'Station and heading maintained for target dwell');return;}
 }else if(ch.type==='resilience'){
  // Automatic mission reports completed waypoints through wpGuidance;
  // manual operator can satisfy each sequential waypoint by entering its radius.
  if(ch.pilot==='auto')ch.completed=Math.min(ch.objectives,s.wpIndex);
  else{const wp=s.waypoints[ch.index];if(wp&&dist(b,wp)<4.5){ch.completed++;ch.index++;logEvent('OBJECTIVE Route checkpoint '+ch.completed);}}
  if(ch.completed===ch.objectives&&ch.gnssTriggered&&ch.lossObserved&&ch.recovered){endChallenge(ch,true,'Route complete with supervised GNSS/C2 recovery');return;}
  // If route finishes before injected events, operator must resume with a longer scenario;
  // avoid a misleading success without observing both faults.
 }
 const target=challengeGoal(ch);
 let err=0;
 if(ch.type==='channel'){
  const g=ch.gates[Math.min(ch.index,ch.gates.length-1)];if(g)err=Math.abs((b.x-g.x)*(-g.uy)+(b.y-g.y)*g.ux);
 }else if(ch.type==='inspection'){
  const g=ch.targets[ch.index];if(g)err=Math.abs(dist(b,g)-9);
 }else if(ch.type==='station')err=dist(b,ch.station);
 else err=s.xte==null?Math.min(15,dist(b,target)):Math.abs(s.xte);
 if(Number.isFinite(err)){ch.accuracySum+=Math.min(30,err)*dt;ch.accuracyN+=dt;}
 ch.metrics=challengeMetric(ch);ch.score=ch.metrics.total;ch.prior={x:b.x,y:b.y};
}
function updateChallengeDisplay(){
 const ch=challenge;
 if(!ch){$('challengeState').textContent='NOT STARTED';$('challengeScore').textContent='0 / 100';$('challengeProgress').textContent='0 / 0';$('challengeSafety').textContent='100';$('challengeAccuracy').textContent='—';$('challengeClock').textContent='0.0 s';$('challengeResult').hidden=true;return;}
 $('challengeState').textContent=ch.status.toUpperCase();$('challengeScore').textContent=ch.score+' / 100';
 $('challengeProgress').textContent=ch.completed+' / '+ch.objectives;
 $('challengeSafety').textContent=nfmt(ch.metrics.safety,0);
 $('challengeAccuracy').textContent=nfmt(ch.metrics.tracking,0);
 $('challengeClock').textContent=nfmt(ch.elapsed,1)+' s';
 $('challengeBrief').textContent=challengeBriefs[ch.type];
 const extra=ch.type==='channel'?'Gate '+Math.min(ch.index+1,ch.objectives)+' / '+ch.objectives+' · sequential crossing, no contact':
   ch.type==='inspection'?'Illumination dwell '+nfmt(ch.dwell,1)+' / '+ch.dwellRequired+' s · lamp, range, heading and speed required':
   ch.type==='station'?'Position/heading hold '+nfmt(ch.dwell,1)+' / '+ch.dwellRequired+' s · real ground speed evaluated':
   'GNSS '+(ch.gnssTriggered?'injected':'pending')+' · C2 '+(ch.c2Triggered?'injected':'pending')+' · operator recovery '+(ch.recovered?'acknowledged':'pending');
 $('challengeInstructions').textContent=extra;
 $('challengeResult').hidden=ch.status==='running';
 if(ch.status!=='running')$('challengeResult').textContent=(ch.status==='completed'?'MISSION ACCOMPLISHED':'MISSION UNSUCCESSFUL')+' · '+ch.outcome+' · '+ch.score+'/100 points · '+
   'Objectives '+ch.completed+'/'+ch.objectives+' · Safety '+nfmt(ch.metrics.safety,0)+' · Tracking '+nfmt(ch.metrics.tracking,0)+' · Energy '+nfmt(ch.metrics.energy,0)+' · Timing '+nfmt(ch.metrics.timing,0);
}
function exportChallenge(){
 if(!challenge){notice('Start a challenge first.');return;}
 const ch=challenge,payload={application:'BLUEVECTOR',category:'illustrative USV mission challenge',scenario:{type:ch.type,level:ch.level,pilot:ch.pilot,seed:cfg.seed},
 status:ch.status,elapsed_s:ch.elapsed,completed:ch.completed,objectives:ch.objectives,score:ch.score,subscores:ch.metrics,
 safety:{contacts:s.collisions,warnings:ch.safetyWarnings,failsafe:s.failsafe},observations:ch.observations,
 model:{physics:cfg.physicsMode,estimator:cfg.navFilter},navigation:{waypoints:s.waypoints},events:eventLog};
 download('bluevector_challenge_result.json',JSON.stringify(payload,null,2),'application/json');notice('Challenge result exported.');
}
// ---- END USV MISSION CHALLENGES ----

// Low-amplitude horizontal wave disturbance shared by all physics modes.
// In 6-DOF, an additional regular free-surface wave excites heave/roll/pitch.
function waveForce(b,time){
 const A=waveAmplitude||0,phase=1.4*time+0.065*b.x+0.04*b.y;
 const fx=0.25*A*Math.cos(phase*1.15),fy=A*Math.sin(phase);
 const c=Math.cos(b.psi),sn=Math.sin(b.psi);
 return {X:c*fx+sn*fy,Y:-sn*fx+c*fy,N:0.12*A*Math.sin(phase+0.7)};
}
function windForce(b){
 const c=Math.cos(b.psi),sn=Math.sin(b.psi);
 const cf=currentAt(b.x,b.y,s.t);const vgx=c*b.u-sn*b.v+cf.x, vgy=sn*b.u+c*b.v+cf.y;
 const wx=wind.x-vgx,wy=wind.y-vgy,mag=Math.hypot(wx,wy);
 const fx=0.60*mag*wx,fy=0.60*mag*wy; // Air density, windage and drag lumped into coefficient.
 return {X:c*fx+sn*fy,Y:-sn*fx+c*fy,N:0.12*(-sn*fx+c*fy)};
}
function groundVelocity(b){const flow=currentAt(b.x,b.y,s.t);
 if(cfg.physicsMode==='6dof'){
   const v=window.Bluevector6DOF.bodyToWorld(b.attitude,[b.u,b.v,b.w]);
   return {x:v[0]+flow.x,y:v[1]+flow.y};
 }
 const c=Math.cos(b.psi),si=Math.sin(b.psi);
 return {x:c*b.u-si*b.v+flow.x,y:si*b.u+c*b.v+flow.y};}
// Physics modes share the same hull geometry, inputs, environmental forces, and state.
// Advanced mode uses a symmetric sway-yaw inertia coupling and damping, plus
// first-order motor response. Illustrative coefficients: NOT vessel calibrated.
function stepVessel(b,cmd,dt){
 // Constrained 2-thruster control allocation: B maps [port,starboard] to
 // [surge force,yaw moment] in BODY axes (x forward, y port, z up).
 const cutoff=!!b.batteryCutoff||b.energy>=hull.batteryWh*(1-(hull.powertrain.cutoffSoc??.02));
 const intentSurge=2*hull.maxThrust*cmd.throttle;
 const intentYaw=-2*hull.arm*hull.maxThrust*cmd.steer;
 const weights=allocationPriority==='heading'?{surgeWeight:.6,yawWeight:1.5}:
  allocationPriority==='surge'?{surgeWeight:1.5,yawWeight:.6}:{};
 const allocated=LA.allocate(intentSurge,intentYaw,hull.arm,hull.maxThrust,
  {limitFraction:cutoff?0:1,regularization:1e-11,...weights});
 allocatedControl=allocated;
 const wantPort=allocated.port,wantStar=allocated.starboard;
 const lag=cfg.physicsMode!=='baseline'?1-Math.exp(-dt/hull.motorTau):1;
 b.motorPort+=lag*(wantPort-b.motorPort);b.motorStar+=lag*(wantStar-b.motorStar);
 // Deliver only thrust supported by the motor and actual battery bus.
 // Power is based on propulsor axial advance velocities, not thrust alone.
 const electric=window.BluevectorPowertrain.evaluate(b.motorPort,b.motorStar,b,hull,hull.hotelPower+(lamp?12:0),dt);
 const port=electric.port,star=electric.star;
 const wf=windForce(b),wv=waveForce(b,s.t);
 const X=port+star-hull.Xu*b.u-hull.Xuu*Math.abs(b.u)*b.u+wf.X+wv.X;
 const Y=-hull.Yv*b.v-hull.Yvv*Math.abs(b.v)*b.v+wf.Y+wv.Y;
 const N=hull.arm*(star-port)-hull.Nr*b.r-hull.Nrr*Math.abs(b.r)*b.r+wf.N+wv.N;
 // Optional six-axis solver: same motor outputs and environmental forces,
 // but 6 rigid-body velocities + quaternion attitude are authoritative.
 if(cfg.physicsMode==='6dof'){
  const flow=currentAt(b.x,b.y,s.t);
  window.Bluevector6DOF.step(b,{hull,time:s.t,dt,wave:{height:cfg.waveHeight,period:cfg.wavePeriod,bearing:cfg.waveBearing},
   current:flow,currentDerivatives:currentDerivativesAt(b.x,b.y,s.t),
   force:{X:port+star+wf.X+wv.X,Y:wf.Y+wv.Y,N:hull.arm*(star-port)+wf.N+wv.N,
   M:(hull.sixDof?.thrusterZ??-.26)*(port+star)}},dt);
 }else{
 let du,dv,dr;
 if(cfg.physicsMode==='advanced'){
  // Matrix M=[[m11,0,0],[0,m22,m23],[0,m23,m33]].
  // Matrix C is skew-symmetric; damping cross-terms are symmetric.
  const h=hull.m22*b.v+hull.m23*b.r;
  const rhsU=X+h*b.r;
  const rhsV=Y-hull.m11*b.u*b.r-hull.Dvr*b.r;
  const rhsN=N-h*b.u+hull.m11*b.u*b.v-hull.Dvr*b.v;
  const det=hull.m22*hull.m33-hull.m23*hull.m23;
  du=rhsU/hull.m11;
  dv=(hull.m33*rhsV-hull.m23*rhsN)/det;
  dr=(hull.m22*rhsN-hull.m23*rhsV)/det;
 }else{
  du=(X+hull.m22*b.v*b.r)/hull.m11;
  dv=(Y-hull.m11*b.u*b.r)/hull.m22;
  dr=(N-(hull.m22-hull.m11)*b.u*b.v)/hull.m33;
 }
 b.u=clamp(b.u+du*dt,-3.5,5.0);b.v=clamp(b.v+dv*dt,-2.5,2.5);
 b.r=clamp(b.r+dr*dt,-1.5,1.5);b.psi=wrap(b.psi+b.r*dt);
 if(cfg.physicsMode!=='6dof'){
  const g=groundVelocity(b);b.x+=g.x*dt;b.y+=g.y*dt;b.dist+=Math.hypot(g.x,g.y)*dt;
 }
 }
 b.port=port;b.star=star;
 b.rpmPort=Math.sign(port)*Math.sqrt(Math.abs(port)/hull.maxThrust)*3400;
 b.rpmStar=Math.sign(star)*Math.sqrt(Math.abs(star)/hull.maxThrust)*3400;
 b.power=electric.power;
 b.usefulPower=electric.usefulPower;b.shaftPower=electric.shaftPower;
 b.batteryLoss=electric.batteryHeat;b.batteryDischarge=electric.chemicalPower;
 b.powertrainScale=electric.scale;b.batteryCutoff=electric.batteryCutoff;
 b.dcVoltage=electric.busVoltage;
 const motorHeat=electric.portMotor.losses+electric.starMotor.losses;
 b.motorTemp+=((motorHeat)-hull.motorCooling*(b.motorTemp-20))*dt/hull.motorThermalCapacity;
 // Energy represents depletion of chemical energy, including internal cell losses.
 b.energy=Math.min(hull.batteryWh*(1-(hull.powertrain.cutoffSoc??.02)),b.energy+electric.chemicalPower*dt/3600);
}
// Convert noisy body-frame gyro readings to ZYX heading rate.
// Roll/pitch are noisy attitude observations, NOT a full 3-axis INS estimate.
// This prevents treating body r as Euler yaw-dot during six-axis motion.
function observedYawRate(dt){
 const b=s.boat;
 const gyroR=b.r+cfg.gyroBiasTrue+normal()*.005/Math.sqrt(dt);
 if(cfg.physicsMode!=='6dof')return gyroR;
 const gyroQ=b.q+normal()*.003/Math.sqrt(dt);
 const rollObs=b.roll+normal()*rad(.4);
 const pitchObs=b.pitch+normal()*rad(.4);
 return (gyroQ*Math.sin(rollObs)+gyroR*Math.cos(rollObs))/Math.max(.1,Math.cos(pitchObs));
}
// A 4-state EKF: [E, N, yaw, gyro bias]. GNSS updates are 2 Hz;
// compass 5 Hz; gyro integration 60 Hz. Process/measurement noise is
// illustrative and documented; covariance is predicted and updated.
function transpose(A){return A[0].map((_,j)=>A.map(row=>row[j]));}
function mul(A,B){return A.map(row=>B[0].map((_,j)=>row.reduce((s,x,k)=>s+x*B[k][j],0)));}
function matAdd(A,B){return A.map((row,i)=>row.map((x,j)=>x+B[i][j]));}
function ekfScalar(e,index,z,R,angular=false){
 const val=[e.x,e.y,e.psi,e.gyroBias];
 const innovation=angular?wrap(z-val[index]):z-val[index];
 const S=e.P[index][index]+R;
 const K=e.P.map(row=>row[index]/S);
 for(let i=0;i<4;i++)val[i]+=K[i]*innovation;
 e.x=val[0];e.y=val[1];e.psi=wrap(val[2]);e.gyroBias=val[3];
 const I=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>(i===j?1:0)-(j===index?K[i]:0)));
 const A=mul(mul(I,e.P),transpose(I));
 e.P=A.map((row,i)=>row.map((v,j)=>v+K[i]*R*K[j]));
}
function stepEKF(dt){
 const e=s.est,b=s.boat;
 const gyroMeasured=observedYawRate(dt);
 const gyroCorrected=gyroMeasured-e.gyroBias;
 const measuredU=b.u+normal()*.035,measuredV=b.v+normal()*.035;
 const yaw=e.psi,c=Math.cos(yaw),sn=Math.sin(yaw);
 const flow=currentAt(e.x,e.y,s.t);
 const vx=c*measuredU-sn*measuredV+flow.x*.82+.025;
 const vy=sn*measuredU+c*measuredV+flow.y*.82-.014;
 e.x+=vx*dt;e.y+=vy*dt;e.psi=wrap(e.psi+gyroCorrected*dt);
 const F=[[1,0,(-sn*measuredU-c*measuredV)*dt,0],
          [0,1,(c*measuredU-sn*measuredV)*dt,0],
          [0,0,1,-dt],[0,0,0,1]];
 const Q=[.018*dt,.018*dt,.000075*dt,rad(.012)**2*dt];
 e.P=matAdd(mul(mul(F,e.P),transpose(F)),Q.map((x,i)=>Q.map((_,j)=>i===j?x:0)));
 e.compassClock+=dt;
 if(e.compassClock>=.2-1e-8){e.compassClock-=.2;
  const measurement=sampleMagneticHeading();
  const uncertaintyDeg=magSettings.calibrated?1.8:
    3.0+.35*magSettings.hardIron_uT+.10*magSettings.softIron_percent;
  ekfScalar(e,2,measurement,rad(uncertaintyDeg)**2,true);
 }
 e.gnssClock+=dt;
 if(s.gnss&&e.gnssClock>=.5-1e-8){e.gnssClock-=.5;
  const outlier=rng()<cfg.gnssOutlier;
  const mx=b.x+normal()*cfg.gnssSigma+(outlier?(rng()<.5?-1:1)*14:0);
  const my=b.y+normal()*cfg.gnssSigma+(outlier?(rng()<.5?-1:1)*12:0);
  const R=cfg.gnssSigma*cfg.gnssSigma;
  // Full 2×2 innovation covariance, rather than assuming x/y independence.
  const a=e.P[0][0]+R,d=e.P[1][1]+R,k=e.P[0][1];
  const det=Math.max(1e-12,a*d-k*k);
  const dx=mx-e.x,dy=my-e.y;
  const nis=(d*dx*dx-2*k*dx*dy+a*dy*dy)/det;
  e.nis=nis;
  if(nis<=9.21034037){ // Chi-square 2 DoF 99% threshold
   ekfScalar(e,0,mx,R);ekfScalar(e,1,my,R);
   e.accepted++;e.lastFix=s.t;
  }else{e.rejected++;}
 }
 e.sigma=Math.sqrt(Math.max(0,(e.P[0][0]+e.P[1][1])/2));
}
function stepLegacyEstimator(dt){
 const e=s.est,b=s.boat;
 const yawRate=observedYawRate(dt);
 e.psi=wrap(e.psi+yawRate*dt);
 const u=b.u+normal()*.04,v=b.v+normal()*.04;
 const c=Math.cos(e.psi),sn=Math.sin(e.psi),flow=currentAt(e.x,e.y,s.t);
 e.x+=(c*u-sn*v+flow.x*.82+.025)*dt;
 e.y+=(sn*u+c*v+flow.y*.82-.014)*dt;
 e.compassClock+=dt;
 if(e.compassClock>=.2-1e-8){e.compassClock-=.2;
  const compass=sampleMagneticHeading();e.psi=wrap(e.psi+.14*wrap(compass-e.psi));}
 e.gnssClock+=dt;
 if(s.gnss&&e.gnssClock>=.5-1e-8){e.gnssClock-=.5;
  const mx=b.x+normal()*cfg.gnssSigma,my=b.y+normal()*cfg.gnssSigma;
  e.x=.35*e.x+.65*mx;e.y=.35*e.y+.65*my;e.lastFix=s.t;e.sigma=cfg.gnssSigma;e.accepted++;
 }else{const age=s.t-e.lastFix;e.sigma=clamp(e.sigma+(.085+.012*age)*dt,cfg.gnssSigma,65);}
}
function stepEstimator(dt){
 const beforeGnss=s.est.accepted,beforeMag=magneticSample.seq;
 if(cfg.navFilter==='ekf')stepEKF(dt);else stepLegacyEstimator(dt);
 // Diagnostic 15x15 error-state covariance; NOT a replacement for the 4-state
 // EKF or a claim of a complete inertial/GNSS navigation solution.
 const b=s.boat,e=s.est,q=b.attitude||LA.fromEuler(b.roll||0,b.pitch||0,b.psi);
 e.insCov=LA.covariancePredict(e.insCov,dt,q,[b.p||0,b.q||0,b.r||0]);
 if(e.accepted>beforeGnss){
  const gnssVar=Math.max(.01,cfg.gnssSigma**2);
  e.insCov=LA.covarianceScalarUpdate(e.insCov,0,gnssVar);
  e.insCov=LA.covarianceScalarUpdate(e.insCov,1,gnssVar);
 }
 if(magneticSample.seq!==beforeMag){
  const sigma=rad(magSettings.calibrated?1.8:3.0+.35*magSettings.hardIron_uT+.10*magSettings.softIron_percent);
  e.insCov=LA.covarianceScalarUpdate(e.insCov,8,sigma*sigma);
 }
 e.insMetrics=LA.covarianceMetrics(e.insCov);
}
// Segment-level detour construction around charted static hazards. This is a
// simplified visibility-route heuristic, not a globally optimal path planner.
function planRoute(start,waypoints){
 const route=[];let from={x:start.x,y:start.y};
 for(let index=0;index<waypoints.length;index++){
  const to=waypoints[index],dx=to.x-from.x,dy=to.y-from.y;
  const L=Math.max(Math.hypot(dx,dy),0.001),ux=dx/L,uy=dy/L;
  const hits=[];
  for(const ob of obstacles){
   const along=(ob.x-from.x)*ux+(ob.y-from.y)*uy;
   const lateral=(ob.x-from.x)*(-uy)+(ob.y-from.y)*ux;
   if(along>0&&along<L&&Math.abs(lateral)<ob.r+8){hits.push({ob,along,lateral});}
  }
  hits.sort((a,b)=>a.along-b.along);
  for(const hit of hits){
   const {ob,lateral}=hit,sign=lateral>=0?-1:1,clearance=ob.r+10,lead=ob.r+8;
   const nx=-uy*sign,ny=ux*sign;
   const pre={x:clamp(ob.x-ux*lead+nx*clearance,4,WORLD.w-4),y:clamp(ob.y-uy*lead+ny*clearance,4,WORLD.h-4),virtual:true};
   const post={x:clamp(ob.x+ux*lead+nx*clearance,4,WORLD.w-4),y:clamp(ob.y+uy*lead+ny*clearance,4,WORLD.h-4),virtual:true};
   route.push(pre,post);
  }
  route.push({x:to.x,y:to.y,checkpoint:true,originalIndex:index});
  from={x:to.x,y:to.y};
 }
 return route;
}
function wpGuidance(){
 const e=s.est,pts=s.navRoute;
 if(!pts.length||s.navIndex>=pts.length){s.finished=true;return {psi:e.psi,speed:0};}
 let target=pts[s.navIndex];let d=dist(e,target);
 if(d<(target.virtual?5.3:4.5)){
   if(target.checkpoint){s.wpIndex=target.originalIndex+1;logEvent('Waypoint '+s.wpIndex+' reached');}
   else logEvent('Hazard detour passed');
   s.navIndex++;s.segmentStart={x:target.x,y:target.y};
   if(s.navIndex>=pts.length){s.finished=true;paused=!(challenge&&challenge.status==='running'&&challenge.type==='channel');refreshButtons();logEvent('MISSION COMPLETE');notice('Mission complete. Simulation paused for inspection; this is not an active station-keeping controller.');return {psi:e.psi,speed:0};}
   target=pts[s.navIndex];d=dist(e,target);
 }
 s.wpDist=d;
 const prev=s.segmentStart;
 const vx=target.x-prev.x,vy=target.y-prev.y,L=Math.max(1e-9,Math.hypot(vx,vy));
 const along=clamp(((e.x-prev.x)*vx+(e.y-prev.y)*vy)/L,0,L);
 s.xte=(vx*(e.y-prev.y)-vy*(e.x-prev.x))/L;
 const look=clamp(7+Math.abs(s.boat.u)*3,7,17);
 const trackTo=clamp(along+look,0,L);
 const los={x:prev.x+vx*trackTo/L,y:prev.y+vy*trackTo/L};
 const aim=dist(los,e)<2?target:los;
 return {psi:Math.atan2(aim.y-e.y,aim.x-e.x),speed:Math.min(2.65,Math.max(0.65,d*0.33))};
}
// Reactive geometric avoidance is a teaching aid, NOT a COLREGs implementation.
function avoidance(headingRad,speed){
 const e=s.est,b=s.boat,course=vec(Math.cos(headingRad),Math.sin(headingRad));
 let modifier=0,slow=1,minRisk=Infinity;
 for(const h of obstacles){
  const dx=h.x-e.x,dy=h.y-e.y,d=Math.hypot(dx,dy);minRisk=Math.min(minRisk,d-h.r-2.5);
  const forward=dx*course.x+dy*course.y;
  const cross=course.x*dy-course.y*dx;
  const safe=h.r+5.5;
  if(forward>-4&&forward<19&&Math.abs(cross)<safe){
    const factor=clamp((25-forward)/25,0,1)*clamp((safe-Math.abs(cross))/safe,0,1);
    modifier+=(cross>=0?-1:1)*0.38*factor; // Turn AWAY from obstacle (signed in ENU).
    slow=Math.min(slow,0.68);
  }
  if(d<h.r+5)slow=Math.min(slow,0.18);
 }
 if(s.traffic){
  const t=s.traffic,g=groundVelocity(b);
  const rx=t.x-b.x,ry=t.y-b.y,rvx=t.vx-g.x,rvy=t.vy-g.y;
  const tv=rvx*rvx+rvy*rvy;
  const tcpa=tv>0.002?clamp(-(rx*rvx+ry*rvy)/tv,0,18):0;
  const dcpa=Math.hypot(rx+rvx*tcpa,ry+rvy*tcpa);
  s.risk='CPA '+nfmt(dcpa,1)+' m';
  if(tcpa<14&&dcpa<9&&Math.hypot(rx,ry)<38){
   const cross=course.x*ry-course.y*rx;
   modifier+=(cross>=0?-1:1)*0.82;
   slow=Math.min(slow,0.55);
  }
 }else s.risk=nfmt(Math.max(0,minRisk),1)+' m clr';
 return {psi:wrap(headingRad+clamp(modifier,-1.15,1.15)),speed:speed*slow};
}
function commandFromControl(dt){
 const b=s.boat;
 s.xte=null;s.wpDist=null;s.desiredSpeed=0;
 if(!s.c2){s.lossTime+=dt;if(s.lossTime>=1.5)setFailsafe('C2 link lost');}
 else s.lossTime=0;
 if(s.mode!=='manual'&&(!s.gnss)&&(s.t-s.est.lastFix>15))setFailsafe('GNSS unavailable > 15 s');
 if(b.batteryCutoff||b.energy>=hull.batteryWh*(1-(hull.powertrain.cutoffSoc??.02)))setFailsafe('Usable battery depleted');
 if(s.failsafe||s.finished)return {throttle:0,steer:0};
 if(s.mode==='manual'){
  if(!s.c2)return {throttle:0,steer:0};
  let throttle=0,steer=0;
  if(helmMode==='joystick'&&joy.active){throttle=-joy.y;steer=joy.x;}
  else{throttle=(keys.ArrowUp?1:0)-(keys.ArrowDown?0.6:0);steer=(keys.ArrowRight?1:0)-(keys.ArrowLeft?1:0);}
  s.desiredSpeed=throttle*3;
  return {throttle,steer};
 }
 if(s.mode==='heading'){
  const psi=psiOfHeading(s.targetHeading);return headingController(psi,2.0);
 }
 if(challenge&&challenge.status==='running'&&challenge.pilot==='auto'&&['inspection','station'].includes(challenge.type))return challengeCommand(challenge);
 const goal=wpGuidance();if(s.finished)return {throttle:0,steer:0};
 const safe=avoidance(goal.psi,goal.speed);
 return headingController(safe.psi,safe.speed);
}
function headingController(desiredPsi,targetSpeed){
 const b=s.boat,e=s.est;
 const err=wrap(desiredPsi-e.psi);
 const steering=clamp(-1.3*err+0.85*b.r,-0.78,0.78);
 let sp=targetSpeed;
 if(Math.abs(err)>1.0)sp=Math.min(sp,0.75);
 const throttle=clamp(0.62*(sp-b.u),-0.50,0.95);
 s.desiredSpeed=sp;
 return {throttle,steer:steering};
}
function checkContact(){
 const b=s.boat;
 let contact=null,nx=0,ny=0;
 for(const ob of obstacles){const d=dist(b,ob),threshold=ob.r+2.2;if(d<threshold){contact=ob;const q=Math.max(d,0.001);nx=(b.x-ob.x)/q;ny=(b.y-ob.y)/q;b.x=ob.x+nx*threshold;b.y=ob.y+ny*threshold;break;}}
 // Training gate posts and inspection targets are physical contact obstacles only
 // while their challenge is active. Keep the original free-simulation chart unchanged.
 if(!contact&&challenge?.status==='running'&&challenge.type==='channel'){
  for(const gate of challenge.gates){for(const post of [gate.port,gate.star]){
   const threshold=hull.beam/2+0.7;
   if(dist(b,post)<threshold){contact={name:'Channel marker contact'};const d=Math.max(.001,dist(b,post));nx=(b.x-post.x)/d;ny=(b.y-post.y)/d;b.x=post.x+nx*threshold;b.y=post.y+ny*threshold;break;}
  }if(contact)break;}
 }
 if(!contact&&challenge?.status==='running'&&challenge.type==='inspection'){
  for(const target of challenge.targets){const threshold=hull.length/2+0.8;
   if(dist(b,target)<threshold){contact={name:'Inspection marker contact'};const d=Math.max(.001,dist(b,target));nx=(b.x-target.x)/d;ny=(b.y-target.y)/d;b.x=target.x+nx*threshold;b.y=target.y+ny*threshold;break;}
  }
 }
 if(!contact&&s.traffic){const ob=s.traffic,d=dist(b,ob),threshold=ob.r+2.2;if(d<threshold){contact={name:'Traffic contact'};const q=Math.max(d,0.001);nx=(b.x-ob.x)/q;ny=(b.y-ob.y)/q;b.x=ob.x+nx*threshold;b.y=ob.y+ny*threshold;}}
 if(!contact&&(b.x<2.2||b.x>WORLD.w-2.2||b.y<2.2||b.y>WORLD.h-2.2)){
  contact={name:'Operating-area boundary'};b.x=clamp(b.x,2.2,WORLD.w-2.2);b.y=clamp(b.y,2.2,WORLD.h-2.2);
 }
 if(contact&&!s.failsafe){s.collisions++;b.u=0;b.v=0;b.r=0;b.w=0;b.p=0;b.q=0;setFailsafe('Contact: '+contact.name);}
}
function update(dt){
 const previous={x:s.boat.x,y:s.boat.y};
 s.t+=dt;
 if(s.traffic){const t=s.traffic;t.x+=t.vx*dt;t.y+=t.vy*dt;if(t.y>WORLD.h-10||t.y<9)t.vy*=-1;}
 const cmd=commandFromControl(dt);
 stepVessel(s.boat,cmd,dt);
 stepEstimator(dt);
 checkContact();
 updateChallenge(dt,previous);
 if(s.xte!=null)s.maxXte=Math.max(s.maxXte,Math.abs(s.xte));
 const eNav=dist(s.est,s.boat);s.navErrorSum+=eNav*eNav;s.navErrorN++;
 s.sampleClock+=dt;s.traceClock+=dt;
 if(s.traceClock>=0.3){s.traceClock-=0.3;s.track.push({x:s.boat.x,y:s.boat.y});if(s.track.length>4000)s.track.shift();}
 if(s.sampleClock>=0.5){s.sampleClock-=0.5;
  const b=s.boat,g=groundVelocity(b);
  s.samples.push({t:Number(s.t.toFixed(2)),x_e_m:b.x,y_n_m:b.y,est_x_m:s.est.x,est_y_m:s.est.y,
   heading_deg:heading(b.psi),surge_mps:b.u,sway_mps:b.v,sog_mps:Math.hypot(g.x,g.y),speed_target_mps:s.desiredSpeed,
   heave_z_m:b.z,heave_w_mps:b.w,roll_deg:deg(b.roll),pitch_deg:deg(b.pitch),
   roll_rate_dps:deg(b.p),pitch_rate_dps:deg(b.q),wave_surface_m:b.waveElevation||0,
   attitude_quat:cfg.physicsMode==='6dof'?JSON.stringify(b.attitude):'',
   yaw_dps:deg(b.r),port_N:b.port,starboard_N:b.star,power_W:b.power,energy_Wh:b.energy,
   battery_pct:100*Math.max(0,1-b.energy/(hull.batteryWh*(1-(hull.powertrain.cutoffSoc??.02)))),battery_cutoff:!!b.batteryCutoff,sigma_pos_m:s.est.sigma,
   nav_error_m:dist(s.est,b),gnss_nis:s.est.nis,gnss_accepted:s.est.accepted,gnss_rejected:s.est.rejected,
   xte_m:s.xte,motor_port_rpm:b.rpmPort,motor_star_rpm:b.rpmStar,dc_voltage_V:b.dcVoltage,motor_temp_C:b.motorTemp,
   hydro_displaced_volume_m3:b.displacedVolume??null,mechanical_energy_J:b.mechanicalEnergy??null,
   wave_steepness_limit:!!b.waveModelLimit,hydrostatic_limit_events:b.hydrostaticLimitEvents||0,
   physics_validity:b.physicsValidity||'ILLUSTRATIVE_NOT_VALIDATED',
   useful_propulsion_W:b.usefulPower||0,shaft_power_W:b.shaftPower||0,
   battery_internal_loss_W:b.batteryLoss||0,battery_chemical_discharge_W:b.batteryDischarge||0,
   thrust_derating_factor:b.powertrainScale??1,physics_guardrail_count:b.limitEvents||0,
   challenge_type:challenge?.type||'',challenge_status:challenge?.status||'',challenge_score:challenge?.score??null,challenge_objectives:challenge?.completed??null,
   physics_mode:cfg.physicsMode,nav_filter:cfg.navFilter,mode:s.mode,gnss:s.gnss,c2:s.c2,wp_index:s.wpIndex,mission_complete:s.finished,failsafe:s.failsafe||''});
  if(s.samples.length>9000)s.samples.shift();
 }
}
function drawMap(){
 ctx.clearRect(0,0,CW,CH);
 const sea=ctx.createLinearGradient(0,0,CW,CH);sea.addColorStop(0,'#0b3657');sea.addColorStop(0.55,'#123d61');sea.addColorStop(1,'#0a2745');ctx.fillStyle=sea;ctx.fillRect(0,0,CW,CH);
 // Chart grid and soundings are non-georeferenced educational graphics.
 ctx.lineWidth=1;ctx.strokeStyle='#8ed8e313';for(let x=0;x<=WORLD.w;x+=10){ctx.beginPath();ctx.moveTo(toX(x),0);ctx.lineTo(toX(x),CH);ctx.stroke();}
 for(let y=0;y<=WORLD.h;y+=10){ctx.beginPath();ctx.moveTo(0,toY(y));ctx.lineTo(CW,toY(y));ctx.stroke();}
 ctx.fillStyle='#9bcee260';ctx.font='12px system-ui';for(let x=20;x<WORLD.w;x+=20)ctx.fillText(x+'m E',toX(x)+3,CH-13);
 for(let y=20;y<WORLD.h;y+=20)ctx.fillText(y+'m N',6,toY(y)-3);
 // Chart north marker.
 ctx.fillStyle='#d8f8ff';ctx.font='bold 18px system-ui';ctx.fillText('N ↑',CW-60,34);
 // Shallow zones.
 for(const ob of obstacles){
  const x=toX(ob.x),y=toY(ob.y),r=ob.r*SCALE;
  ctx.fillStyle='#eac36d17';ctx.strokeStyle='#ffe0a277';ctx.lineWidth=2;ctx.setLineDash([8,6]);
  ctx.beginPath();ctx.arc(x,y,r,0,TAU);ctx.fill();ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#dfcb8b';ctx.font='11px system-ui';ctx.fillText('⚠ '+ob.name,x+r+4,y);
  ctx.strokeStyle='#f9bd7580';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-7,y-7);ctx.lineTo(x+7,y+7);ctx.moveTo(x-7,y+7);ctx.lineTo(x+7,y-7);ctx.stroke();
 }
 // Current vector visualization.
 const arrow=(x,y,vx,vy,color)=>{ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+vx,y+vy);ctx.stroke();const a=Math.atan2(vy,vx);ctx.beginPath();ctx.moveTo(x+vx,y+vy);ctx.lineTo(x+vx-9*Math.cos(a-.55),y+vy-9*Math.sin(a-.55));ctx.lineTo(x+vx-9*Math.cos(a+.55),y+vy-9*Math.sin(a+.55));ctx.closePath();ctx.fill();};
 for(let x=90;x<CW;x+=220)for(let y=110;y<CH;y+=180){arrow(x,y,current.x*28,-current.y*28,'#67e5e64a');}
 // Training mission objects: symbolic markers only, not collision bodies or surveyed chart data.
 if(challenge){const ch=challenge;
  if(ch.type==='channel')ch.gates.forEach((gate,i)=>{
   const active=i===ch.index,passed=i<ch.completed;
   ctx.strokeStyle=passed?'#83f2bf':active?'#73eaff':'#9cbdd5';ctx.lineWidth=3;ctx.setLineDash(active?[4,4]:[]);
   ctx.beginPath();ctx.moveTo(toX(gate.port.x),toY(gate.port.y));ctx.lineTo(toX(gate.star.x),toY(gate.star.y));ctx.stroke();ctx.setLineDash([]);
   for(const pos of [gate.port,gate.star]){ctx.fillStyle=passed?'#39ae8b':'#e3b05e';ctx.beginPath();ctx.arc(toX(pos.x),toY(pos.y),6,0,TAU);ctx.fill();}
   ctx.fillStyle='#ecfcff';ctx.font='bold 13px system-ui';ctx.fillText('GATE '+(i+1),toX(gate.x)+8,toY(gate.y)-9);
  });
  if(ch.type==='inspection')ch.targets.forEach((target,i)=>{
    const passed=i<ch.completed;ctx.strokeStyle=passed?'#6af9b8':'#ffd37b';ctx.lineWidth=2;ctx.setLineDash([5,5]);
    ctx.beginPath();ctx.arc(toX(target.x),toY(target.y),9*SCALE,0,TAU);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=passed?'#9ef0c7':'#ffe4a0';ctx.font='bold 13px system-ui';ctx.fillText('INSPECT '+(i+1),toX(target.x)+5,toY(target.y)-9);
  });
  if(ch.type==='station'){const target=ch.station;ctx.strokeStyle='#6af0ed';ctx.lineWidth=2;ctx.setLineDash([6,6]);
    ctx.beginPath();ctx.arc(toX(target.x),toY(target.y),ch.radius*SCALE,0,TAU);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#dffaff';ctx.font='bold 14px system-ui';ctx.fillText('STATION HOLD',toX(target.x)+9,toY(target.y)-10);
  }
 }
 // Mission route, starting from assigned segment origin.
 if(s.waypoints.length){
  ctx.lineWidth=3;ctx.setLineDash([7,7]);ctx.strokeStyle='#f0cf7fa8';ctx.beginPath();let prev={x:21,y:48};ctx.moveTo(toX(prev.x),toY(prev.y));
  const activeRoute=s.mode==='auto'?s.navRoute:(s.previewRoute||s.waypoints);
  for(const wp of activeRoute){ctx.lineTo(toX(wp.x),toY(wp.y));}ctx.stroke();ctx.setLineDash([]);
  if(s.mode==='auto'||s.previewRoute)for(const wp of activeRoute.filter(w=>w.virtual)){ctx.fillStyle='#c596ff';ctx.beginPath();ctx.arc(toX(wp.x),toY(wp.y),4,0,TAU);ctx.fill();}
  s.waypoints.forEach((wp,i)=>{
   const reached=i<s.wpIndex;
   ctx.beginPath();ctx.arc(toX(wp.x),toY(wp.y),reached?9:12,0,TAU);ctx.fillStyle=reached?'#185c55':'#214ea1';ctx.fill();ctx.strokeStyle=reached?'#92f4cb':'#c7e9ff';ctx.lineWidth=2;ctx.stroke();
   ctx.fillStyle='#fff';ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='bold 11px system-ui';ctx.fillText(''+(i+1),toX(wp.x),toY(wp.y));ctx.textAlign='left';ctx.textBaseline='alphabetic';
  });
 }
 // Draw estimated track.
 if(s.track.length>1){ctx.beginPath();ctx.moveTo(toX(s.track[0].x),toY(s.track[0].y));for(let i=1;i<s.track.length;i++)ctx.lineTo(toX(s.track[i].x),toY(s.track[i].y));ctx.strokeStyle='#70e9ffb5';ctx.lineWidth=2.5;ctx.stroke();}
 // Traffic target and velocity vector.
 if(s.traffic){const t=s.traffic,x=toX(t.x),y=toY(t.y);ctx.save();ctx.translate(x,y);ctx.rotate(-Math.atan2(t.vy,t.vx));ctx.fillStyle='#ff9b85';ctx.strokeStyle='#ffe3da';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(18,0);ctx.lineTo(-13,9);ctx.lineTo(-9,0);ctx.lineTo(-13,-9);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();
  ctx.fillStyle='#ffc5b7';ctx.font='11px system-ui';ctx.fillText('AIS TEST CONTACT',x+19,y-13);
  arrow(x,y,t.vx*40,-t.vy*40,'#ffc5b7aa');
 }
 const b=s.boat,est=s.est;
 // Estimated position and one sigma ring. Different from truth.
 const ex=toX(est.x),ey=toY(est.y);
 ctx.strokeStyle='#ffd580ab';ctx.fillStyle='#ffd58013';ctx.lineWidth=2;ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(ex,ey,clamp(est.sigma*SCALE,4,170),0,TAU);ctx.fill();ctx.stroke();ctx.setLineDash([]);
 ctx.strokeStyle='#ffd88d';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(ex-8,ey);ctx.lineTo(ex+8,ey);ctx.moveTo(ex,ey-8);ctx.lineTo(ex,ey+8);ctx.stroke();
 const bx=toX(b.x),by=toY(b.y);
 // Velocity vector: SOG in ENU.
 const g=groundVelocity(b);arrow(bx,by,24*g.x,-24*g.y,'#80f6ffad');
 // Vessel model: royal-blue twin-thruster USV drawn in body coordinates (bow +x).
 ctx.save();ctx.translate(bx,by);ctx.rotate(-b.psi);
 const L=hull.length*SCALE, B=hull.beam*SCALE;
 // Wake illustrates true travel path, independent from heading.
 ctx.strokeStyle='#a0edff63';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(-L*.55,-B*.36);ctx.lineTo(-L*1.10,-B*.65);ctx.moveTo(-L*.55,B*.36);ctx.lineTo(-L*1.10,B*.65);ctx.stroke();
 ctx.fillStyle='#142946';ctx.strokeStyle='#b6dfff';ctx.lineWidth=2;
 ctx.beginPath();ctx.moveTo(L*.56,0);ctx.quadraticCurveTo(L*.34,-B*.54,-L*.46,-B*.45);ctx.lineTo(-L*.57,-B*.2);ctx.lineTo(-L*.57,B*.2);ctx.lineTo(-L*.46,B*.45);ctx.quadraticCurveTo(L*.34,B*.54,L*.56,0);ctx.closePath();ctx.fill();ctx.stroke();
 ctx.fillStyle='#4169e1';ctx.fillRect(-L*.23,-B*.31,L*.48,B*.62);
 ctx.fillStyle='#b1d8ff';ctx.fillRect(-L*.12,-B*.21,L*.20,B*.42);
 ctx.fillStyle='#173f78';ctx.fillRect(-L*.49,-B*.32,L*.18,B*.20);ctx.fillRect(-L*.49,B*.12,L*.18,B*.20);
 ctx.fillStyle='#eff9ff';ctx.beginPath();ctx.arc(L*.42,0,3.2,0,TAU);ctx.fill();
 if(lamp){const grd=ctx.createLinearGradient(L*.4,0,L*7,0);grd.addColorStop(0,'#9dceff65');grd.addColorStop(1,'#82adff00');ctx.fillStyle=grd;ctx.beginPath();ctx.moveTo(L*.4,0);ctx.lineTo(L*7,-L*1.65);ctx.lineTo(L*7,L*1.65);ctx.closePath();ctx.fill();}
 ctx.restore();
 // Scale bar bottom-left.
 ctx.strokeStyle='#e2f6ff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(22,CH-32);ctx.lineTo(22+10*SCALE,CH-32);ctx.moveTo(22,CH-38);ctx.lineTo(22,CH-26);ctx.moveTo(22+10*SCALE,CH-38);ctx.lineTo(22+10*SCALE,CH-26);ctx.stroke();ctx.fillStyle='#fff';ctx.font='11px system-ui';ctx.fillText('10 m',26,CH-40);
 if(s.failsafe){ctx.fillStyle='#490d1ad9';ctx.fillRect(CW*.27,CH*.06,CW*.46,49);ctx.strokeStyle='#ff91a0';ctx.strokeRect(CW*.27,CH*.06,CW*.46,49);ctx.fillStyle='#fff';ctx.font='bold 21px system-ui';ctx.textAlign='center';ctx.fillText('FAILSAFE: IDLE / DRIFT',CW*.5,CH*.06+31);ctx.textAlign='left';}
 else if(challenge&&challenge.status!=='running'){
   ctx.fillStyle=challenge.status==='completed'?'#0b4c40eb':'#572236ec';ctx.fillRect(CW*.24,CH*.055,CW*.52,60);
   ctx.fillStyle='#fff';ctx.font='bold 22px system-ui';ctx.textAlign='center';
   ctx.fillText((challenge.status==='completed'?'MISSION ACCOMPLISHED':'MISSION UNSUCCESSFUL')+' · '+challenge.score+'/100',CW*.5,CH*.055+39);ctx.textAlign='left';
 }
 else if(s.finished){ctx.fillStyle='#0a4b3fdc';ctx.fillRect(CW*.30,CH*.06,CW*.4,48);ctx.fillStyle='#c5ffdc';ctx.font='bold 20px system-ui';ctx.textAlign='center';ctx.fillText('MISSION COMPLETE',CW*.5,CH*.06+30);ctx.textAlign='left';}
 if(planning){ctx.fillStyle='#062c49cc';ctx.fillRect(CW*.33,CH*.9,CW*.34,35);ctx.fillStyle='#e6f9ff';ctx.font='bold 15px system-ui';ctx.textAlign='center';ctx.fillText('TAP TO ADD WAYPOINT',CW*.5,CH*.9+23);ctx.textAlign='left';}
}
function drawTrend(){
 tctx.clearRect(0,0,500,170);tctx.fillStyle='#071b30';tctx.fillRect(0,0,500,170);
 for(let y=25;y<170;y+=39){tctx.strokeStyle='#285273';tctx.lineWidth=1;tctx.beginPath();tctx.moveTo(35,y);tctx.lineTo(485,y);tctx.stroke();}
 tctx.font='15px system-ui';tctx.fillStyle='#9cc4de';tctx.fillText('m/s',7,17);
 const data=s.samples.slice(-120);if(data.length<2)return;
 const plot=(fn,color)=>{tctx.strokeStyle=color;tctx.lineWidth=3;tctx.beginPath();data.forEach((v,i)=>{const x=35+i/(Math.max(1,data.length-1))*448,y=150-clamp(fn(v),0,4.5)*27;i?tctx.lineTo(x,y):tctx.moveTo(x,y);});tctx.stroke();};
 plot(v=>v.sog_mps,'#5ce8f7');plot(v=>v.speed_target_mps,'#ffc978');
 tctx.fillStyle='#5ce8f7';tctx.fillRect(301,12,11,4);tctx.fillStyle='#c3dfed';tctx.fillText('SOG',317,20);tctx.fillStyle='#ffc978';tctx.fillRect(381,12,11,4);tctx.fillStyle='#c3dfed';tctx.fillText('SET',397,20);
}
function display(){
 const b=s.boat,g=groundVelocity(b),sog=Math.hypot(g.x,g.y),bat=clamp(100*(1-b.energy/(hull.batteryWh*(1-(hull.powertrain.cutoffSoc??.02)))),0,100);
 const hdg=(Math.round(heading(b.psi))%360).toString().padStart(3,'0')+'°';
 $('sog').textContent=nfmt(sog,2);$('mapSpeed').textContent=nfmt(sog,1);
 $('hdg').textContent=hdg;$('mapHeading').textContent=hdg;
 $('stw').textContent=nfmt(Math.hypot(b.u,b.v),2);$('battery').textContent=nfmt(bat,1)+'%';
 $('vel').textContent=nfmt(b.u,2)+' / '+nfmt(b.v,2);$('yawrate').textContent=nfmt(deg(b.r),1)+' °/s';
 $('heave').textContent=nfmt(b.z,2)+' m / '+nfmt(b.w,2)+' m/s';
 $('attitude').textContent=nfmt(deg(b.roll),1)+'° / '+nfmt(deg(b.pitch),1)+'°';
 $('attitudeRate').textContent=nfmt(deg(b.p),1)+' / '+nfmt(deg(b.q),1)+' °/s';
 $('waveReadout').textContent=nfmt(b.waveElevation||0,2)+' m'+(cfg.physicsMode==='6dof'?' · '+nfmt(cfg.waveHeight,2)+' m H':' · 3-DOF');
 const proxy=window.Bluevector6DOF.hydrostatics(hull);$('hydroReadout').textContent=cfg.physicsMode==='6dof'?nfmt(proxy.draft_m,3)+' / '+nfmt(proxy.GMt_m,2)+' m':'6-DOF only';
 $('motors').textContent=Math.round(b.port)+' / '+Math.round(b.star)+' N';$('power').textContent=Math.round(b.power)+' W';
 $('energyAudit').textContent=Math.round(b.usefulPower||0)+' / '+Math.round(b.batteryLoss||0)+' W';
 $('physicsGuard').textContent=(100*(b.powertrainScale??1)).toFixed(0)+'% / '+(b.limitEvents||0)+(b.hydrostaticLimitEvents?' · HYDRO LIMIT':'')+(b.waveModelLimit?' · STEEP WAVE':'');
 $('cog').textContent=sog>0.08?String(Math.round(heading(Math.atan2(g.y,g.x)))).padStart(3,'0')+'°':'—';
 $('rpm').textContent=Math.round(b.rpmPort)+' / '+Math.round(b.rpmStar);
 $('electrical').textContent=nfmt(b.dcVoltage,1)+' V / '+nfmt(b.motorTemp,1)+'°C';
 const label=s.failsafe?'FAILSAFE':challenge?.status==='completed'?'CHALLENGE COMPLETE':challenge?.status==='failed'?'CHALLENGE FAILED':s.finished?'COMPLETE':s.mode==='auto'?'AUTO WP':s.mode==='heading'?'HDG HOLD':'MANUAL';
 $('activeMode').textContent=label;
 $('xte').textContent=s.xte==null?'—':nfmt(s.xte,1)+' m';
 $('wpDist').textContent=s.wpDist==null?'—':nfmt(s.wpDist,1)+' m';
 $('uncert').textContent=nfmt(s.est.sigma,1)+' m (1σ)';
 $('gnssAge').textContent=nfmt(s.t-s.est.lastFix,1)+' s';$('risk').textContent=s.risk;
 $('filterStatus').textContent=(cfg.navFilter==='ekf'?'EKF':'LEGACY')+' · '+(s.est.nis==null?'—':nfmt(s.est.nis,1))+' NIS';
 $('navError').textContent=nfmt(dist(s.est,b),2)+' m';
 $('gnssHealth').textContent=s.gnss?'GNSS NOMINAL':'GNSS OUTAGE';$('gnssHealth').style.color=s.gnss?'#99f4cb':'#ffcf88';
 $('c2Health').textContent=s.c2?'C2 LINK OK':'C2 LOST';$('c2Health').style.color=s.c2?'#99f4cb':'#ffcf88';
 $('targetHdg').textContent=String(Math.round(s.targetHeading)%360).padStart(3,'0')+'°';
 $('mapWP').textContent=challenge?(challenge.completed+' / '+challenge.objectives):(s.wpIndex+' / '+s.waypoints.length);$('progress').textContent=challenge?(challenge.completed+' / '+challenge.objectives):(s.wpIndex+' / '+s.waypoints.length);
 $('energy').textContent=nfmt(b.energy,1)+' Wh';$('distance').textContent=nfmt(b.dist,1)+' m';$('collisions').textContent=s.collisions;
 const t=Math.floor(s.t);$('timeText').textContent='T+ '+String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0');
 $('statusText').textContent=paused?'PAUSED':label;
 $('statusDot').style.background=s.failsafe?'#ff7f92':paused?'#ffd479':s.finished?'#72e5ad':'#76e6b4';
 $('lamp').classList.toggle('selected',lamp);$('lamp').textContent=lamp?'💡 Lamp ON':'💡 Lamp';
 $('sound').textContent=muted?'🔇 Muted':'🔊 Sound';
 $('resume').disabled=!s.failsafe;
 drawTrend();
 if(activeWorkspace==='challenge')updateChallengeDisplay();
 if(activeWorkspace==='analysis'&&performance.now()-lastAnalysisDraw>180){drawAnalysis();lastAnalysisDraw=performance.now();}
}
function refreshButtons(){
 document.querySelectorAll('[data-mode]').forEach(el=>el.classList.toggle('selected',el.dataset.mode===s.mode));
 $('gnss').textContent=s.gnss?'GNSS ON':'GNSS OFF';$('gnss').classList.toggle('warn',!s.gnss);
 $('c2').textContent=s.c2?'C2 ON':'C2 OFF';$('c2').classList.toggle('warn',!s.c2);
 $('plan').classList.toggle('selected',planning);canvas.classList.toggle('planning',planning);
 $('pause').textContent=paused?'▶ Play':'⏸ Pause';
 $('helmMode').textContent=helmMode==='joystick'?'Joystick':'D-pad';
 $('helmSwitch').textContent=helmMode==='joystick'?'Use D-pad':'Use Joystick';
}
function buildHelm(){
 const mount=$('controlMount');mount.replaceChildren();joy.active=false;joy.x=0;joy.y=0;
 if(helmMode==='joystick'){
  const pad=document.createElement('div'),knob=document.createElement('div');pad.className='joypad';knob.className='joyknob';knob.textContent='✦';pad.append(knob);
  const position=e=>{const r=pad.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,max=r.width*.34;let dx=e.clientX-cx,dy=e.clientY-cy;const len=Math.hypot(dx,dy);if(len>max){dx=dx/len*max;dy=dy/len*max;}joy={x:dx/max,y:dy/max,active:true};knob.style.transform='translate('+dx+'px,'+dy+'px)';};
  pad.addEventListener('pointerdown',e=>{e.preventDefault();pad.setPointerCapture(e.pointerId);position(e);});
  pad.addEventListener('pointermove',e=>{if(joy.active)position(e);});
  const up=()=>{joy={x:0,y:0,active:false};knob.style.transform='translate(0,0)';};
  pad.addEventListener('pointerup',up);pad.addEventListener('pointercancel',up);pad.addEventListener('lostpointercapture',up);mount.append(pad);
 }else{
  const dpad=document.createElement('div');dpad.className='dpad';
  const sequence=[null,['▲','ArrowUp'],null,['◀','ArrowLeft'],null,['▶','ArrowRight'],null,['▼','ArrowDown'],null];
  for(const spec of sequence){if(!spec){dpad.append(document.createElement('span'));continue;}const btn=document.createElement('button');btn.textContent=spec[0];btn.setAttribute('aria-label',spec[1]);
   const code=spec[1];btn.addEventListener('pointerdown',e=>{e.preventDefault();btn.setPointerCapture(e.pointerId);keys[code]=true;});
   const up=()=>keys[code]=false;btn.addEventListener('pointerup',up);btn.addEventListener('pointercancel',up);btn.addEventListener('lostpointercapture',up);dpad.append(btn);
  }
  mount.append(dpad);
 }
 refreshButtons();
}
function download(name,text,mime){const blob=new Blob([text],{type:mime});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportCSV(){if(!s.samples.length){notice('Run the scenario briefly before exporting telemetry.');return;}
 const fields=Object.keys(s.samples[0]);const value=v=>v==null?'':typeof v==='string'?'"'+v.replaceAll('"','""')+'"':String(v);
 const lines=[fields.join(','),...s.samples.map(row=>fields.map(k=>value(row[k])).join(','))];download('bluevector_usv_telemetry.csv',lines.join('\r\n'),'text/csv;charset=utf-8');notice('Telemetry CSV exported.');}
function exportJSON(){const payload={description:'Illustrative non-validated USV digital twin scenario',config:cfg,hull,environment:{current,wind},obstacles,waypoints:s.waypoints,
 challenge:challenge?{type:challenge.type,level:challenge.level,pilot:challenge.pilot,status:challenge.status,score:challenge.score,completed:challenge.completed,outcome:challenge.outcome}:null,
 state:{t:s.t,mode:s.mode,boat:s.boat,estimate:s.est,wpIndex:s.wpIndex,completed:s.finished,failsafe:s.failsafe,collisions:s.collisions,simRate},traffic:s.traffic,samples:s.samples,events:eventLog};
 download('bluevector_usv_mission.json',JSON.stringify(payload,null,2),'application/json');notice('Scenario JSON exported.');}
function drawAnalysis(){
 const data=s.samples.slice(-650),c=actx,W=analysisCanvas.width,H=analysisCanvas.height;
 c.clearRect(0,0,W,H);c.fillStyle='#061b2f';c.fillRect(0,0,W,H);
 const kind=$('analyticTrace').value;let series=[];
 if(kind==='speed')series=[{key:'sog_mps',name:'SOG m/s',color:'#5ce8f7'},{key:'speed_target_mps',name:'Target m/s',color:'#ffd479'}];
 if(kind==='error')series=[{key:'nav_error_m',name:'True error m',color:'#ffce76'},{key:'sigma_pos_m',name:'1σ component m',color:'#5ce8f7'}];
 if(kind==='power')series=[{key:'power_W',name:'Power W',color:'#5ce8f7'},{key:'motor_temp_C',name:'Motor °C',color:'#ffce76'}];
 if(kind==='heading')series=[{key:'heading_deg',name:'Heading °',color:'#5ce8f7'},{key:'yaw_dps',name:'Yaw °/s',color:'#ffce76'}];
 if(kind==='attitude')series=[{key:'roll_deg',name:'Roll °',color:'#5ce8f7'},{key:'pitch_deg',name:'Pitch °',color:'#ffce76'}];
 if(kind==='heave')series=[{key:'heave_z_m',name:'Heave m',color:'#5ce8f7'},{key:'wave_surface_m',name:'Surface m',color:'#ffce76'}];
 const left=58,right=W-25,top=35,bottom=H-38;
 c.strokeStyle='#264d6f';c.lineWidth=1;c.font='13px system-ui';c.fillStyle='#a9cbe7';
 for(let k=0;k<=4;k++){const y=top+(bottom-top)*k/4;c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();}
 if(data.length<2){c.fillText('Run the mission to populate telemetry traces.',left+15,H/2);return;}
 const values=series.flatMap(se=>data.map(row=>Number(row[se.key])).filter(Number.isFinite));
 let mn=Math.min(...values,0),mx=Math.max(...values,1);const padding=Math.max(.1,(mx-mn)*.08);mn-=padding;mx+=padding;
 const tx=t=>left+(t-data[0].t)/Math.max(.01,data[data.length-1].t-data[0].t)*(right-left);
 const ty=v=>bottom-(v-mn)/(mx-mn)*(bottom-top);
 for(let j=0;j<=4;j++){const v=mx-(mx-mn)*j/4;c.fillText(v.toFixed(1),6,top+(bottom-top)*j/4+4);}
 for(const se of series){c.strokeStyle=se.color;c.lineWidth=2.6;c.beginPath();let started=false;
  for(const row of data){const v=Number(row[se.key]);if(!Number.isFinite(v))continue;
    if(!started){c.moveTo(tx(row.t),ty(v));started=true;}else c.lineTo(tx(row.t),ty(v));}c.stroke();}
 c.fillStyle='#b9d4e7';c.fillText('T + '+data[0].t.toFixed(0)+' s',left,bottom+23);
 c.fillText('T + '+data[data.length-1].t.toFixed(0)+' s',right-90,bottom+23);
 series.forEach((se,i)=>{c.fillStyle=se.color;c.fillRect(left+i*240,11,13,5);c.fillStyle='#d4e8f6';c.fillText(se.name,left+19+i*240,18);});
 $('aRmse').textContent=nfmt(Math.sqrt(s.navErrorSum/Math.max(1,s.navErrorN)),2);
 $('aXte').textContent=nfmt(s.maxXte,2);
 $('aNis').textContent=s.est.accepted+' / '+s.est.rejected;
 $('aEndurance').textContent=nfmt(Math.max(0,hull.batteryWh*(1-(hull.powertrain.cutoffSoc??.02))-s.boat.energy)/Math.max(1,s.boat.power)*60,0);
}
function setWorkspace(name){
 activeWorkspace=name;$('shell').dataset.workspace=name;
 document.querySelectorAll('button[data-workspace]').forEach(btn=>btn.classList.toggle('selected',btn.dataset.workspace===name));
 if(name==='twin')setView('3d');else if(name==='mission')setView('2d');else if(name==='helm')setView('both');else if(name==='challenge')setView('2d');
 if(name==='analysis')drawAnalysis();
}
function loadPreset(kind){
 reset();if(kind==='calm'){current.x=0;current.y=0;wind.x=0;wind.y=0;waveAmplitude=0;notice('Calm water preset loaded.');}
 if(kind==='wind'){current.x=.28;current.y=-.12;wind.x=0;wind.y=9;waveAmplitude=8;notice('Crosswind disturbance loaded.');}
 if(kind==='denied'){current.x=.3;current.y=.08;waveAmplitude=10;notice('GNSS challenge loaded. Disable GNSS during autopilot to test resilience.');}
 logEvent('Preset loaded: '+kind);
}
function safeRouteCheck(){
 if(challenge?.status==='running'){notice('Challenge objectives are locked.');return;}
 if(!s.waypoints.length){notice('Add mission waypoints first.');return;}
 const route=planRoute(s.est,s.waypoints);let clearance=Infinity,from={x:s.est.x,y:s.est.y};
 for(const to of route){for(const ob of obstacles){
  const dx=to.x-from.x,dy=to.y-from.y,L2=Math.max(1e-9,dx*dx+dy*dy);
  const t=clamp(((ob.x-from.x)*dx+(ob.y-from.y)*dy)/L2,0,1);
  clearance=Math.min(clearance,Math.hypot(ob.x-from.x-t*dx,ob.y-from.y-t*dy)-ob.r);
 }from=to;}
 s.previewRoute=route;
 const verdict=clearance>=5.5?'GEOMETRIC CLEARANCE OK':'CAUTION: clearance below 5.5 m';
 notice('Preview '+route.length+' nodes · minimum segment clearance '+nfmt(clearance,1)+' m · '+verdict+'. No dynamic-traffic or COLREG safety guarantee.');
}
function importScenario(file){
 if(challenge?.status==='running'){notice('Mission imports are locked during challenges.');return;}
 if(!file)return;const reader=new FileReader();reader.onload=()=>{
  try{const obj=JSON.parse(reader.result);if(!Array.isArray(obj.waypoints)||!obj.waypoints.length||obj.waypoints.length>18)throw Error('Expected 1–18 waypoints');
   const pts=obj.waypoints.map(p=>({x:Number(p.x),y:Number(p.y)}));
   if(pts.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x<=2||p.x>=WORLD.w-2||p.y<=2||p.y>=WORLD.h-2||obstacles.some(h=>dist(p,h)<h.r+5)))throw Error('Waypoint outside permitted demonstration area or within hazard clearance');
   reset();s.waypoints=pts;s.previewRoute=null;notice('Imported '+pts.length+' mission waypoints. Press Waypoint auto to begin.');logEvent('Mission file imported');
  }catch(e){notice('Mission import rejected: '+e.message);}finally{$('importFile').value='';}
 };reader.readAsText(file);
}
function resume(){if(!s.failsafe)return;if(!s.c2){notice('Restore C2 link first.');return;}if(s.mode!=='manual'&&(!s.gnss||s.t-s.est.lastFix>2)){notice('Autonomous restart requires a recent GNSS fix.');return;}
 s.failsafe=null;s.lossTime=0;logEvent('Failsafe cleared by operator');notice('Control resumed.');}
window.addEventListener('keydown',e=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code))e.preventDefault();if(e.repeat)return;
 if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code))keys[e.code]=true;
 if(e.code==='KeyT'){lampToggle=!lampToggle;lamp=lampToggle||lampHeld;feedback('lamp');}
 if(e.code==='Space'){paused=!paused;refreshButtons();}
 if(e.code==='KeyR')reset();});
window.addEventListener('keyup',e=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code))keys[e.code]=false;});
window.addEventListener('blur',()=>{keys=Object.create(null);joy.active=false;joy.x=0;joy.y=0;});
document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>toggleMode(btn.dataset.mode)));
$('plan').addEventListener('click',()=>{if(challenge?.status==='running'){notice('Waypoint editing is locked during challenge.');return;}planning=!planning;refreshButtons();notice(planning?'Tap chart to place waypoints.':'Waypoint planning disabled.');});
canvas.addEventListener('pointerdown',e=>{if(challenge?.status==='running'||!planning)return;e.preventDefault();const r=canvas.getBoundingClientRect();const x=clamp((e.clientX-r.left)/r.width*WORLD.w,3,WORLD.w-3),y=clamp((1-(e.clientY-r.top)/r.height)*WORLD.h,3,WORLD.h-3);
 if(obstacles.some(o=>Math.hypot(o.x-x,o.y-y)<o.r+5)){notice('Waypoint too close to a charted hazard.');return;}
 if(s.waypoints.length>=18){notice('Limit: 18 waypoints.');return;}
 s.previewRoute=null;s.waypoints.push({x,y});haptic(12);if(s.mode==='auto'){s.mode='manual';s.wpIndex=0;refreshButtons();notice('Route edited. Select Waypoint auto to restart.');}logEvent('Waypoint added at E '+nfmt(x,1)+', N '+nfmt(y,1));});
$('demo').addEventListener('click',()=>{if(challenge?.status==='running'){notice('Route locked during challenge.');return;}s.previewRoute=null;s.waypoints=demoWaypoints();s.wpIndex=0;s.navIndex=0;s.navRoute=planRoute(s.est,s.waypoints);s.segmentStart={x:s.est.x,y:s.est.y};s.finished=false;logEvent('Demo route loaded');});
$('clear').addEventListener('click',()=>{if(challenge?.status==='running'){notice('Route locked during challenge.');return;}s.previewRoute=null;s.waypoints=[];s.wpIndex=0;s.finished=false;s.mode='manual';planning=true;refreshButtons();notice('Route cleared. Tap chart to add waypoints.');});
$('helmSwitch').addEventListener('click',()=>{helmMode=helmMode==='joystick'?'dpad':'joystick';keys=Object.create(null);buildHelm();});
let lampPressStart=0;
$('lamp').addEventListener('pointerdown',e=>{lampPressStart=performance.now();lampHeld=true;lamp=lampToggle||lampHeld;unlockAudio();});
const releaseLamp=e=>{if(!lampHeld)return;const held=performance.now()-lampPressStart;lampHeld=false;if(held<260)lampToggle=!lampToggle;lamp=lampToggle||lampHeld;feedback('lamp');};
$('lamp').addEventListener('pointerup',releaseLamp);
$('lamp').addEventListener('pointercancel',()=>{lampHeld=false;lamp=lampToggle;});
$('lamp').addEventListener('pointerleave',e=>{if(e.buttons){lampHeld=false;lamp=lampToggle;}});
$('lamp').addEventListener('click',e=>{if(e.detail===0){lampToggle=!lampToggle;lamp=lampToggle;feedback('lamp');}});
$('sound').addEventListener('click',()=>{muted=!muted;if(!muted){unlockAudio();chirp(580,.08);}refreshButtons();});
$('gnss').addEventListener('click',()=>{s.gnss=!s.gnss;s.est.gnssClock=0;logEvent(s.gnss?'GNSS restored':'GNSS signal lost');refreshButtons();});
$('c2').addEventListener('click',()=>{s.c2=!s.c2;logEvent(s.c2?'C2 signal restored':'C2 signal lost');refreshButtons();});
$('stop').addEventListener('click',()=>setFailsafe('Operator commanded idle'));
$('resume').addEventListener('click',resume);
$('pause').addEventListener('click',()=>{paused=!paused;refreshButtons();});
$('reset').addEventListener('click',reset);
$('hdgMinus').addEventListener('click',()=>applyHeadingCommand(s.targetHeading-15,'navigation panel'));
$('hdgPlus').addEventListener('click',()=>applyHeadingCommand(s.targetHeading+15,'navigation panel'));
$('csv').addEventListener('click',exportCSV);$('json').addEventListener('click',exportJSON);
$('analysisCSV').addEventListener('click',exportCSV);$('analysisJSON').addEventListener('click',exportJSON);
document.querySelectorAll('button[data-workspace]').forEach(b=>b.addEventListener('click',()=>setWorkspace(b.dataset.workspace)));
$('timeScale').addEventListener('change',e=>{simRate=Number(e.target.value)||1;notice('Simulation rate '+simRate+'× (physics step remains 1/60 s).');});
$('analyticTrace').addEventListener('change',drawAnalysis);
$('undoWp').addEventListener('click',()=>{if(challenge?.status==='running'){notice('Route locked during challenge.');return;}if(!s.waypoints.length)return;s.previewRoute=null;const wp=s.waypoints.pop();s.mode='manual';s.finished=false;s.wpIndex=0;refreshButtons();notice('Removed last waypoint at E '+nfmt(wp.x,1));});
$('safeRoute').addEventListener('click',safeRouteCheck);
$('importTrigger').addEventListener('click',()=>$('importFile').click());
$('importFile').addEventListener('change',e=>importScenario(e.target.files?.[0]));
$('presetCalm').addEventListener('click',()=>loadPreset('calm'));
$('presetWind').addEventListener('click',()=>loadPreset('wind'));
$('presetDenied').addEventListener('click',()=>loadPreset('denied'));
$('challengeKind').addEventListener('change',e=>{if(challenge?.status==='running')return;$('challengeBrief').textContent=challengeBriefs[e.target.value];});
$('challengeLaunch').addEventListener('click',()=>startChallenge($('challengeKind').value,$('challengeLevel').value,$('challengePilot').value));
$('challengeRestart').addEventListener('click',()=>{const c=challenge;startChallenge(c?.type||$('challengeKind').value,c?.level||$('challengeLevel').value,c?.pilot||$('challengePilot').value);});
$('challengeExport').addEventListener('click',exportChallenge);
$('modelTag').textContent=(cfg.physicsMode==='6dof'?'WAVE-COUPLED 6-DOF':cfg.physicsMode==='advanced'?'COUPLED 3-DOF':'BASELINE 3-DOF')+' · '+(cfg.navFilter==='ekf'?'EKF':'LEGACY NAV');
// ========================= BLUEVECTOR 3D PLOTLY TWIN =========================
// Plotly shows the SAME authoritative 3-DOF or 6-DOF state as the 2D chart. The local ENU viewport
// follows the vessel. No independently integrated 3D position or yaw exists.
const twin = {
  view: '3d', camera: 'chase', renderer:'plotly', ready: false, loading: false, pending: false,
  active: true, lastDraw: 0, cameraSwitch: false, plot: $('twinPlot'),
  layers: {thrust:true, axes:true, environment:true, estimate:true, route:true, wake:true, wireframe:false}
};
const modelInfo = {
  'Hull shell':'4.0 m deck with twin trapezoidal pontoons generated from the hydrostatic proxy parameters. Above-water styling and equipment are conceptual.',
  'Deck / roof':'Conceptual deck and cabin geometry; not a certified or CAD-derived hull.',
  'Royal-blue sheer stripe':'BLUEVECTOR hull identity and stylized raised sheer band.',
  'Cabin glazing':'Conceptual camera/helm housing; not an operative vision model.',
  'Sensor hardware':'Stylized GNSS, compass and camera housings. No simulated camera imagery or radar returns.',
  'Twin thruster housings':'Separate port and starboard drives; dynamics use actual signed differential thrust.',
  'Hull edges':'Wireframe overlay for geometry inspection. Not finite-element or structural analysis.'
};
const meshColors = {
  'Hull shell':'#12375E','Deck / roof':'#D8E8FC','Royal-blue sheer stripe':'#4169E1',
  'Cabin glazing':'#103A69','Sensor hardware':'#AACEE8','Twin thruster housings':'#153D75'
};
function buildVesselGeometry(){
 const groups={};const edges=[];
 const group=name=>(groups[name]??={verts:[],i:[],j:[],k:[]});
 const triangle=(name,a,b,c)=>{const g=group(name),idx=g.verts.length;g.verts.push(a,b,c);g.i.push(idx);g.j.push(idx+1);g.k.push(idx+2);};
 const quad=(name,a,b,c,d)=>{triangle(name,a,b,c);triangle(name,a,c,d);};
 const poly=(name,pts)=>{for(let i=1;i<pts.length-1;i++)triangle(name,pts[0],pts[i],pts[i+1]);};
 const box=(name,x0,x1,y0,y1,z0,z1)=>{
   const a=[x0,y0,z0],b=[x1,y0,z0],c=[x1,y1,z0],d=[x0,y1,z0];
   const e=[x0,y0,z1],f=[x1,y0,z1],g=[x1,y1,z1],h=[x0,y1,z1];
   quad(name,a,b,c,d);quad(name,e,f,g,h);quad(name,a,b,f,e);
   quad(name,b,c,g,f);quad(name,c,d,h,g);quad(name,d,a,e,h);
 };
 const cylinder=(name,x,y,z0,z1,r,n=10)=>{
  for(let j=0;j<n;j++){
   const a=TAU*j/n,b=TAU*(j+1)/n;
   const p=[x+r*Math.cos(a),y+r*Math.sin(a),z0],q=[x+r*Math.cos(b),y+r*Math.sin(b),z0];
   const t=[x+r*Math.cos(a),y+r*Math.sin(a),z1],u=[x+r*Math.cos(b),y+r*Math.sin(b),z1];
   quad(name,p,q,u,t);triangle(name,[x,y,z0],q,p);triangle(name,[x,y,z1],t,u);
  }
 };
 const dome=(name,x,y,z,rx,ry,rz)=>{
  const bands=4,N=10;for(let j=0;j<bands;j++){
   const a=Math.PI*j/bands,b=Math.PI*(j+1)/bands;
   for(let k=0;k<N;k++){
    const c=TAU*k/N,d=TAU*(k+1)/N;
    const point=(p,h)=>[x+rx*Math.sin(p)*Math.cos(h),y+ry*Math.sin(p)*Math.sin(h),z+rz*Math.cos(p)];
    quad(name,point(a,c),point(a,d),point(b,d),point(b,c));
   }
  }
 };
 const path=pts=>edges.push(pts);
 // Visible deck outline: dry structure bridging the two actual pontoons.
 // Unlike V7, the decorative shell below the waterline has been removed.
 // All submerged pontoon geometry comes from the SAME hydroGeometry parameters
 // that determine displacement, draft, waterplane inertia and righting stiffness.
 const stations=[[-2.0,.48],[-1.67,.70],[-.9,.78],[0,.80],[.8,.71],[1.43,.45],[1.92,.09],[2.0,.015]];
 for(let j=0;j<stations.length-1;j++){
  const [x0,w0]=stations[j],[x1,w1]=stations[j+1];
  for(const side of [-1,1]){
   const t0=[x0,side*w0,.21],t1=[x1,side*w1,.21];
   const lower0=[x0,side*w0*.96,.06],lower1=[x1,side*w1*.96,.06];
   quad('Hull shell',t0,t1,lower1,lower0);
   quad('Royal-blue sheer stripe',[x0,side*w0*.995,.18],[x1,side*w1*.995,.18],
     [x1,side*w1*.97,.11],[x0,side*w0*.97,.11]);
   quad('Deck / roof',[x0,0,.21],[x1,0,.21],t1,t0);
  }
 }
 // Two independent trapezoidal pontoons, x-aligned with the vessel.
 // At the waterline z=0, each has width b; at keel z=-draft each
 // has width (2*Cb-1)*b. This cross-section integrates to V=Awp*d*Cb.
 const hp=window.Bluevector6DOF.hydrostatics(hull);
 const geom=hull.sixDof.hydroGeometry, hl=geom.waterlineLength/2;
 const halfTop=geom.pontoonWidth/2,halfKeel=halfTop*hp.bottomWidthRatio;
 for(const side of [-1,1]){
  const center=side*geom.pontoonCenterOffset;
  const x0=-hl,x1=hl;
  const a=[x0,center-halfTop,0],b=[x1,center-halfTop,0],
        c=[x1,center+halfTop,0],d=[x0,center+halfTop,0];
  const e=[x0,center-halfKeel,-hp.draft_m],f=[x1,center-halfKeel,-hp.draft_m],
        g=[x1,center+halfKeel,-hp.draft_m],h=[x0,center+halfKeel,-hp.draft_m];
  const ra=[x0,center-halfTop,.19],rb=[x1,center-halfTop,.19],
        rc=[x1,center+halfTop,.19],rd=[x0,center+halfTop,.19];
  quad('Hull shell',a,b,f,e);quad('Hull shell',d,c,g,h);quad('Hull shell',e,f,g,h);
  quad('Hull shell',a,d,h,e);quad('Hull shell',b,c,g,f);
  quad('Hull shell',a,b,rb,ra);quad('Hull shell',d,c,rc,rd);
  quad('Hull shell',ra,rb,rc,rd);quad('Hull shell',a,d,rd,ra);quad('Hull shell',b,c,rc,rb);
  path([a,b,c,d,a]);path([e,f,g,h,e]);
 }
 // Boxy but proportioned cabin and angled windshield.
 box('Deck / roof',-.90,.46,-.38,.38,.23,.72);
 quad('Cabin glazing',[.46,-.38,.35],[.64,-.35,.49],[.64,.35,.49],[.46,.38,.35]);
 box('Cabin glazing',-.58,.15,-.395,-.365,.40,.64);
 box('Cabin glazing',-.58,.15,.365,.395,.40,.64);
 box('Deck / roof',-.96,.38,-.43,.43,.70,.79);
 box('Cabin glazing',-.24,.36,-.31,.31,.795,.805);
 // Twin propulsor pods, installed near aft quarters.
 for(const side of [-1,1]){
   const z=hull.sixDof.thrusterZ,y=side*hull.arm;
   box('Twin thruster housings',-1.77,-1.49,y-.07,y+.07,z-.045,z+.045);
   cylinder('Sensor hardware',-1.68,y,z-.045,z+.045,.075,8);
 }
 // Radar-look sensor dome, GNSS receiver, mast and electro-optic housing.
 cylinder('Sensor hardware',-.32,0,.78,1.48,.045,10);
 dome('Sensor hardware',-.32,0,1.53,.27,.27,.17);
 box('Sensor hardware',.34,.63,-.10,.10,.80,1.03);
 dome('Cabin glazing',.64,0,.93,.11,.11,.10);
 cylinder('Sensor hardware',-1.05,.17,.27,1.29,.022,8);
 cylinder('Sensor hardware',-1.10,-.18,.27,.94,.023,8);
 cylinder('Sensor hardware',.85,0,.25,.43,.045,10);
 // Hull sheer, cabin roof, bow rail and aft rail polylines.
 for(const side of [-1,1]){
   path(stations.map(([x,w])=>[x,side*w,.23]));
   path([[-.96,side*.43,.8],[.38,side*.43,.8],[.55,side*.37,.53]]);
   path([[-1.93,side*.38,.38],[-1.35,side*.62,.40]]);
 }
 path(stations.map(([x])=>[x,0,.235]));
 return {groups,edges};
}
const vesselGeometry=buildVesselGeometry();
function bodyToENU(point,boat=s.boat){
 const q=boat.attitude||LA.fromEuler(boat.roll||0,boat.pitch||0,boat.psi);
 return LA.transform(q,point,[0,0,boat.z||0]);
}
function xyzTrace(name,localPoints,color,width=4,visible=true,extra={}){
 const x=[],y=[],z=[];
 for(const p of localPoints){if(p===null){x.push(null);y.push(null);z.push(null);}else{x.push(p[0]);y.push(p[1]);z.push(p[2]);}}
 return {type:'scatter3d',mode:'lines',name,x,y,z,line:{color,width},hoverinfo:'name',showlegend:false,visible,...extra};
}
function bodyLines(lines){const points=[];for(const line of lines){for(const p of line)points.push(p===null?null:bodyToENU(p));points.push(null);}return points;}
function arrowLine(a,b){
 const dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
 const len=Math.hypot(dx,dy,dz)||1, ux=dx/len,uy=dy/len;
 const side=[-uy,ux,0],lenHead=Math.min(.32,len*.35);
 return [a,b,null,[b[0]-lenHead*ux+.16*side[0],b[1]-lenHead*uy+.16*side[1],b[2]-lenHead*dz/len],b,
  [b[0]-lenHead*ux-.16*side[0],b[1]-lenHead*uy-.16*side[1],b[2]-lenHead*dz/len]];
}
function waterSurface(){
 const grid=Array.from({length:19},(_,i)=>-9+i);
 const z=grid.map(y=>grid.map(x=>{
  if(cfg.physicsMode==='6dof')return window.Bluevector6DOF.waveAt(s.boat.x+x,s.boat.y+y,s.t,
    {height:cfg.waveHeight,period:cfg.wavePeriod,bearing:cfg.waveBearing}).z;
  return .028*Math.sin(.85*x+.53*y)+.018*Math.cos(1.25*y-.27*x)-.025;
 }));
 return {type:'surface',name:'Stylized water',x:grid,y:grid,z,
   colorscale:[[0,'#071A36'],[.35,'#0B3456'],[.7,'#0B5277'],[1,'#0A7296']],
   showscale:false,opacity:.88,hoverinfo:'skip',showlegend:false,lighting:{ambient:.75,diffuse:.45,specular:.75,roughness:.25}};
}
function makeTwinTraces(){
 const b=s.boat, traces=[waterSurface()],c=Math.cos(b.psi),si=Math.sin(b.psi);
 for(const [name,g] of Object.entries(vesselGeometry.groups)){
  const pts=g.verts.map(p=>bodyToENU(p));
  traces.push({type:'mesh3d',name,x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),
   i:g.i,j:g.j,k:g.k,color:meshColors[name],opacity:1,flatshading:true,
   lighting:{ambient:.70,diffuse:.63,roughness:.62,specular:.38,fresnel:.11},
   lightposition:{x:16,y:-9,z:20},hovertemplate:name+'<extra></extra>',showlegend:false});
 }
 traces.push(xyzTrace('Hull edges',bodyLines(vesselGeometry.edges),'#61D8FF',2,twin.layers.wireframe));
 // Two signed physical propulsion vectors (not force arrows at an invented rudder).
 for(const [side,force,color,nm] of [[1,b.port,'#38e1ff','Port thruster'],[-1,b.star,'#6d92ff','Starboard thruster']]){
  const start=[-1.68,side*hull.arm,hull.sixDof.thrusterZ],end=[-1.68+1.25*force/hull.maxThrust,side*hull.arm,hull.sixDof.thrusterZ];
  traces.push(xyzTrace(nm,bodyLines([arrowLine(start,end)]),color,6,twin.layers.thrust&&Math.abs(force)>1));
 }
 // Moving BODY-fixed axes: X forward, Y to port, Z upwards.
 for(const [nm,to,color] of [['Body X / bow',[3.10,0,.55],'#ff9ca9'],['Body Y / port',[0,2.2,.55],'#8dffc1'],['Body Z / up',[0,0,2.55],'#6faeff']]){
  traces.push(xyzTrace(nm,bodyLines([arrowLine([0,0,.55],to)]),color,3,twin.layers.axes));
 }
 // Wind and current are earth-fixed ENU vectors. No rotation by vessel yaw.
 const ca=[-4.8,-5.1,.2],cb=[ca[0]+current.x*7,ca[1]+current.y*7,.2];
 traces.push(xyzTrace('Water current (ENU)',arrowLine(ca,cb),'#5bf5ed',5,twin.layers.environment));
 const wa=[-4.8,4.7,.25],wb=[wa[0]+wind.x*.40,wa[1]+wind.y*.40,.25];
 traces.push(xyzTrace('Wind TO (ENU)',arrowLine(wa,wb),'#ffd479',5,twin.layers.environment));
 // Vessel truth at origin of local ENU scene; estimated state is relative offset.
 const delta=[s.est.x-b.x,s.est.y-b.y];
 const estimateLine=[[delta[0]-.32,delta[1],.30],[delta[0]+.32,delta[1],.30],null,
  [delta[0],delta[1]-.32,.30],[delta[0],delta[1]+.32,.30]];
 traces.push(xyzTrace('Estimated GNSS/DR position',estimateLine,'#ffdb83',5,twin.layers.estimate));
 const uncertainty=[];const sigma=Math.min(Math.max(s.est.sigma,0.2),8);
 for(let k=0;k<=48;k++){const a=TAU*k/48;uncertainty.push([delta[0]+sigma*Math.cos(a),delta[1]+sigma*Math.sin(a),.04]);}
 traces.push(xyzTrace('Estimated position 1σ ring (illustrative)',uncertainty,'#fbd28b',2,twin.layers.estimate));
 // Waypoint bearing is clipped to the viewport: it is NOT a fake WP coordinate.
 const wp=(challenge&&challenge.status==='running')?challengeGoal(challenge):(s.mode==='auto'?s.navRoute[s.navIndex]:s.waypoints[Math.min(s.wpIndex,s.waypoints.length-1)]);
 if(wp){
  const dx=wp.x-b.x,dy=wp.y-b.y,L=Math.max(.001,Math.hypot(dx,dy));
  const endpoint=[6.7*dx/L,6.7*dy/L,.22];
  traces.push(xyzTrace('Waypoint bearing',arrowLine([0,0,.25],endpoint),'#fce3a4',4,twin.layers.route));
  traces.push({type:'scatter3d',mode:'markers+text',name:'Waypoint direction (range '+L.toFixed(1)+' m)',
   x:[endpoint[0]],y:[endpoint[1]],z:[endpoint[2]],
   text:[(challenge&&challenge.status==='running'?'TARGET':'WP '+(s.wpIndex+1))+' · '+L.toFixed(0)+' m'],textposition:'top center',
   textfont:{color:'#ffe7ac',size:10},marker:{color:'#ffd479',size:4},
   visible:twin.layers.route,showlegend:false,hovertemplate:'Bearing only · waypoint distance '+L.toFixed(1)+' m<extra></extra>'});
 }
 // Nearby training markers use the same local ENU state and remain within the 3D viewport.
 if(challenge&&twin.layers.route){const ch=challenge,near=(p)=>Math.hypot(p.x-b.x,p.y-b.y)<12;
  if(ch.type==='channel'){const g=ch.gates[ch.index];if(g&&(near(g.port)||near(g.star))){
   traces.push({type:'scatter3d',mode:'markers+lines',name:'Next channel gate',
    x:[g.port.x-b.x,g.star.x-b.x],y:[g.port.y-b.y,g.star.y-b.y],z:[.35,.35],
    marker:{size:5,color:'#ffcf78'},line:{color:'#63eaff',width:4},showlegend:false,
    hovertemplate:'Simulated navigation gate<extra></extra>'});
  }}else if(ch.type==='inspection'){const t=ch.targets[ch.index];if(t&&near(t)){
   traces.push({type:'scatter3d',mode:'markers+text',name:'Inspection target',x:[t.x-b.x],y:[t.y-b.y],z:[.45],
    text:['INSPECT'],textposition:'top center',textfont:{color:'#ffdda1'},marker:{size:7,color:'#ffcc6a'},showlegend:false});
  }}else if(ch.type==='station'){const t=ch.station;if(near(t)){
   traces.push({type:'scatter3d',mode:'markers+text',name:'Station hold target',x:[t.x-b.x],y:[t.y-b.y],z:[.25],
    text:['HOLD'],textposition:'top center',textfont:{color:'#a9fff1'},marker:{size:7,color:'#71e7d9'},showlegend:false});
  }}
 }
 // 2D history shown locally; no invented world-coordinate transformation.
 const past=s.track.slice(-160).map(p=>[p.x-b.x,p.y-b.y,.10]).filter(p=>Math.abs(p[0])<12&&Math.abs(p[1])<12);
 traces.push(xyzTrace('Recent simulated track',past,'#5acde0',3,twin.layers.route&&past.length>1));
 // Stylized wakes are a VISUAL effect, not a CFD or 6-DOF seakeeping result.
 const wake=[];const speed=Math.hypot(b.u,b.v);
 if(speed>.25){for(const side of [-1,1]){
  const line=[];for(let k=0;k<17;k++){let t=k/16;
    line.push(bodyToENU([-1.85-5.6*t,side*(.63+1.7*t+.12*Math.sin(10*t)),.06+.04*Math.sin(30*t)]));}
  wake.push(line);
 }}
 traces.push(xyzTrace('Illustrative wake',wake.flatMap(line=>[...line,null]),'#9af2ff',3,twin.layers.wake&&wake.length>0));
 // Unified world-frame Plotly scene: the entire compass ring is earth-fixed;
 // only the cyan/green heading references move with physical vessel/sensor state.
 traces.push(...integratedCompassTraces());
 return traces;
}
// -----------------------------------------------------------------------------
// WORLD-FIXED 3D COMPASS WITH VESSEL AT ITS CENTER
// The ring is a single Plotly scene with the actual USV hull, rather than a
// separately orbiting picture. ENU X=East, Y=North, Z=Up. A magnetic estimate
// cannot modify the physical vessel pose. Clicking rim bearings commands the
// existing heading-hold controller and never rotates the hull instantaneously.
// -----------------------------------------------------------------------------
function integratedCompassTraces(){
 if(!integratedCompass.visible)return [];
 const t=[],n=96,R=integratedCompass.ringRadius,z=integratedCompass.ringHeight;
 const x=[],y=[],zz=[],ii=[],jj=[],kk=[];
 for(let j=0;j<=n;j++){
  const a=TAU*j/n;
  x.push((R-.36)*Math.cos(a),R*Math.cos(a));
  y.push((R-.36)*Math.sin(a),R*Math.sin(a));
  zz.push(z-.07,z-.07);
  if(j<n){const k=2*j;ii.push(k,k+1);jj.push(k+1,k+3);kk.push(k+2,k+2);}
 }
 t.push({type:'mesh3d',name:'True-north compass annulus',x,y,z:zz,i:ii,j:jj,k:kk,
   color:'#255d9b',opacity:.45,flatshading:true,hoverinfo:'skip',showlegend:false});
 const circle=(r,col,width)=>{const p=[];for(let j=0;j<=n;j++){
  const a=j*TAU/n;p.push([r*Math.cos(a),r*Math.sin(a),z]);}
  t.push(xyzTrace('True-north compass ring',p,col,width));};
 circle(R,'#54e8fc',5);circle(R-.38,'#478bbd',2);
 const tick=[];
 for(let h=0;h<360;h+=10){const a=rad(90-h),r0=h%30===0?R-.34:R-.19;
  tick.push([r0*Math.cos(a),r0*Math.sin(a),z+.07],[(R-.07)*Math.cos(a),(R-.07)*Math.sin(a),z+.07],null);
 }
 t.push(xyzTrace('ENU compass degree marks',tick,'#c2e9fc',2));
 const labels=[['N',0,R+.5],['E',R+.5,0],['S',0,-R-.5],['W',-R-.5,0]];
 t.push({type:'scatter3d',mode:'text',name:'True cardinal directions',
  x:labels.map(p=>p[1]),y:labels.map(p=>p[2]),z:labels.map(()=>z+.19),
  text:labels.map(p=>p[0]),textfont:{color:'#f4fdff',size:19,family:'Arial Black, Arial'},
  hoverinfo:'skip',showlegend:false});
 const bow=vesselBowENU(),q=rad(90-s.targetHeading),m=rad(90-magneticSample.estimatedTrueHeading_deg),
  direction=(a)=>[Math.cos(a),Math.sin(a)],d=direction(q),md=direction(m);
 const line=(name,v,from,to,height,col,width)=>{
  t.push(xyzTrace(name,arrowLine([from*v[0],from*v[1],height],[to*v[0],to*v[1],height]),col,width));
 };
 line('Actual USV bow / quaternion truth',bow,2.17,R-.47,z+.19,'#3ff4ff',10);
 line('Commanded heading (USV autopilot target)',d,2.40,R-.63,z+.3,'#ffd37d',6);
 line('Magnetometer estimated heading',md,2.58,R-.80,z+.42,'#86f5b0',5);
 const sigma=s.est.insMetrics?.heading_sigma_deg??3,half=clamp(1.96*sigma,.1,50),sector=[];
 for(let j=0;j<=18;j++){
  const h=magneticSample.estimatedTrueHeading_deg-half+2*half*j/18,a=rad(90-h);
  sector.push([(R-.53)*Math.cos(a),(R-.53)*Math.sin(a),z+.45]);
 }
 t.push(xyzTrace('Diagnostic heading uncertainty sector',sector,'#86f5b0',3));
 const sx=[],sy=[],hd=[];for(let h=0;h<360;h+=10){const a=rad(90-h);
  sx.push((R-.12)*Math.cos(a));sy.push((R-.12)*Math.sin(a));hd.push(h);}
 t.push({type:'scatter3d',mode:'markers',name:'Integrated compass heading command',
  x:sx,y:sy,z:sx.map(()=>z+.21),customdata:hd,
  marker:{size:5,color:'#ffd98b',opacity:.8},showlegend:false,
  hovertemplate:'Set autopilot target to %{customdata}° true<extra></extra>'});
 return t;
}
function cameraPreset(){
 const c=Math.cos(s.boat.psi),sn=Math.sin(s.boat.psi);
 if(twin.camera==='overhead')return {eye:{x:.01,y:.02,z:2.65},up:{x:0,y:1,z:0},center:{x:0,y:0,z:0}};
 if(twin.camera==='orbit')return twin.savedOrbit||{eye:{x:1.38,y:-1.38,z:.95},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};
 return {eye:{x:-1.75*c+.85*sn,y:-1.75*sn-.85*c,z:.92},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};
}
function twinLayout(){
 const camera=twin.camera==='orbit'?cameraPreset():cameraPreset();
 return {paper_bgcolor:'#061b32',plot_bgcolor:'#061b32',margin:{l:0,r:0,t:0,b:0},
   showlegend:false,uirevision:'bluevector-3d-user-camera',
   scene:{bgcolor:'#082239',aspectmode:'manual',aspectratio:{x:1.16,y:1.16,z:.52},
     xaxis:{title:'E local (m)',range:[-7.5,7.5],color:'#9bc5df',gridcolor:'#204e70',zeroline:false,showbackground:false,nticks:5},
     yaxis:{title:'N local (m)',range:[-7.5,7.5],color:'#9bc5df',gridcolor:'#204e70',zeroline:false,showbackground:false,nticks:5},
     zaxis:{title:'U (m)',range:[-1.3,3],color:'#9bc5df',gridcolor:'#204e70',zeroline:false,showbackground:false,nticks:4},
     dragmode:'orbit',camera,uirevision:'bluevector-3d-user-camera'}};
}

// -----------------------------------------------------------------------------
// TRUE NORTH ENU COMPASS / COMMON CAMERA BRIDGE
// No independent compass heading state: true needle = vessel bow vector transformed
// by the exact quaternion used for the 3D hull. Heading command is separate.
// Camera gestures are view-only; commanded headings use differential-thrust GNC.
// -----------------------------------------------------------------------------
const compass={plot:$('compassPlot'),fallback:$('compassFallback'),ready:false,
 loading:false,pending:false,lastDraw:0,linked:true,cameraSwitch:false,
 savedCamera:null,eventsBound:false,detailOpen:false};
// The compass ring is co-located with the USV in the primary Plotly scene.
// Its north direction is earth-fixed, while the cyan bow vector follows
// the vessel's quaternion-derived attitude. A separate diagnostic scene is optional.
const integratedCompass={visible:true,ringRadius:4.15,ringHeight:.36};
const bearing360=v=>((v%360)+360)%360;
function vesselBowENU(){
 const x=bodyToENU([1,0,0]),org=bodyToENU([0,0,0]);
 const dx=x[0]-org[0],dy=x[1]-org[1],r=Math.hypot(dx,dy);
 return r>1e-10?[dx/r,dy/r]:[Math.cos(s.boat.psi),Math.sin(s.boat.psi)];
}
function trueBowDegrees(){
 const q=s.boat.attitude||LA.fromEuler(s.boat.roll||0,s.boat.pitch||0,s.boat.psi);
 const h=LA.headingFromQuaternion(q);
 return h===null?heading(s.boat.psi):h;
}
function compassCamera(){return (twin.camera==='orbit'&&twin.savedOrbit)?structuredClone(twin.savedOrbit):cameraPreset();}
function compass3DLayout(){return {paper_bgcolor:'#071b32',plot_bgcolor:'#071b32',margin:{l:0,r:0,t:0,b:0},
 uirevision:'bluevector-compass-3d',showlegend:false,
 scene:{bgcolor:'#071b32',aspectmode:'manual',aspectratio:{x:1,y:1,z:.26},
 xaxis:{range:[-4.55,4.55],visible:false},yaxis:{range:[-4.55,4.55],visible:false},
 zaxis:{range:[-.55,1.1],visible:false},dragmode:'orbit',camera:compassCamera(),
 uirevision:'bluevector-compass-3d'}};}
function compass3DTraces(){
 const traces=[],outer=3.50,n=96;
 const vx=[0],vy=[0],vz=[-.10],ii=[],jj=[],kk=[];
 for(let i=0;i<=n;i++){const a=TAU*i/n;vx.push(outer*Math.cos(a));vy.push(outer*Math.sin(a));vz.push(-.10);
  if(i<n){ii.push(0);jj.push(i+1);kk.push(i+2);}}
 traces.push({type:'mesh3d',name:'Compass face',x:vx,y:vy,z:vz,i:ii,j:jj,k:kk,
 color:'#102d56',opacity:1,flatshading:true,hoverinfo:'skip',showlegend:false,
 lighting:{ambient:.83,diffuse:.38,specular:.28}});
 const ring=(r,z,c,width=3)=>{const xyz=[];for(let i=0;i<=n;i++){const a=TAU*i/n;xyz.push([r*Math.cos(a),r*Math.sin(a),z]);}
   traces.push(xyzTrace('North-fixed compass ring',xyz,c,width));};
 ring(3.57,.025,'#53dcff',5);ring(3.14,.03,'#467ba9',2);ring(.36,.14,'#67cfff',4);
 const tick=[];for(let a=0;a<360;a+=10){const angle=rad(90-a),r0=a%30===0?2.82:3.00,r1=3.14;
  tick.push([r0*Math.cos(angle),r0*Math.sin(angle),.07],[r1*Math.cos(angle),r1*Math.sin(angle),.07],null);}
 traces.push(xyzTrace('True bearing ticks',tick,'#a9d4ee',2));
 const card=['N','E','S','W'],cx=[0,2.43,0,-2.43],cy=[2.43,0,-2.43,0];
 traces.push({type:'scatter3d',mode:'text',name:'Fixed ENU cardinal directions',x:cx,y:cy,z:[.10,.10,.10,.10],
 text:card,textfont:{color:'#e5f6ff',size:18,family:'Arial Black, Arial'},hoverinfo:'skip',showlegend:false});
 const [nx,ny]=vesselBowENU();
 traces.push(xyzTrace('USV true bow / heading',arrowLine([0,0,.19],[3.0*nx,3.0*ny,.19]),'#40ebff',11));
 // Explicit commanded heading, NOT a claim that the vessel rotates instantaneously.
 const th=rad(90-s.targetHeading);
 traces.push(xyzTrace('Commanded heading',arrowLine([0,0,.25],[2.45*Math.cos(th),2.45*Math.sin(th),.25]),'#ffd37d',5));
 // Realistic sensor estimate is NOT the authoritative true bow. A separate
 // green pointer shows the magnetometer-derived estimate, with a 95% sector
 // from the illustrative 15x15 diagnostic attitude covariance.
 const estimate=bearing360(magneticSample.estimatedTrueHeading_deg),m=rad(90-estimate);
 traces.push(xyzTrace('Magnetometer estimated true heading',
   arrowLine([0,0,.28],[2.12*Math.cos(m),2.12*Math.sin(m),.28]),'#83f2ac',4));
 const sigma=s.est.insMetrics?.heading_sigma_deg??3;
 const half=clamp(1.96*sigma,.1,65),left=rad(90-estimate-half),right=rad(90-estimate+half);
 traces.push(xyzTrace('Approximate 95% heading uncertainty',[
  [0,0,.32],[2.65*Math.cos(left),2.65*Math.sin(left),.32],null,
  ...Array.from({length:19},(_,i)=>{const a=left+(right-left)*i/18;return [2.65*Math.cos(a),2.65*Math.sin(a),.32];}),null,
  [2.65*Math.cos(right),2.65*Math.sin(right),.32],[0,0,.32]
 ],'#83f2ac',2));
 // A click on a rim bearing sets the autopilot heading command (not camera orbit).
 const hx=[],hy=[],hd=[];
 for(let h=0;h<360;h+=10){const a=rad(90-h);hx.push(3.37*Math.cos(a));hy.push(3.37*Math.sin(a));hd.push(h);}
 traces.push({type:'scatter3d',mode:'markers',name:'Click rim to command heading',x:hx,y:hy,z:hx.map(()=>.18),
 customdata:hd,marker:{size:5,color:'#ffcd80',opacity:.62},showlegend:false,
 hovertemplate:'Command heading %{customdata}° true<extra></extra>'});
 return traces;
}
function updateCompassReadouts(){
 const bow=trueBowDegrees(),desired=bearing360(s.targetHeading);
 const fmt=h=>String(Math.round(h)%360).padStart(3,'0')+'°';
 const angularError=Math.abs(((bow-heading(s.boat.psi)+540)%360)-180);
 $('compassCommandValue').textContent=fmt(desired);$('vesselCommandValue').textContent=fmt(desired);
 $('combinedHeadingReadout').textContent='TRUE '+fmt(bow)+' · MAG '+fmt(magneticSample.estimatedTrueHeading_deg)+' · CMD '+fmt(desired);
 // Avoid moving a slider during a drag initiated by the user.
 for(const id of ['compassHeadingInput','vesselHeadingInput']){
  const el=$(id);if(document.activeElement!==el||!el.matches(':active'))el.value=String(Math.round(desired)%360);
 }
 $('compassReadout').textContent='TRUE BOW '+fmt(bow)+' · MAGNETIC EST. '+fmt(magneticSample.estimatedTrueHeading_deg)+
  ' · COMMAND '+fmt(desired)+' · ALIGNMENT '+angularError.toFixed(8)+'°';
 const metrics=s.est.insMetrics||LA.covarianceMetrics(s.est.insCov);
 const measuredError=wrap(rad(90-magneticSample.estimatedTrueHeading_deg)-rad(90-bow));
 $('magneticReadout').textContent='MAGNETOMETER '+(magSettings.calibrated?'CALIBRATED':'UNCALIBRATED')+
   ' · FIELD '+nfmt(magneticSample.fieldNorm_uT,1)+' µT · HEADING '+fmt(magneticSample.estimatedTrueHeading_deg)+
   ' · TRUE ERROR '+nfmt(deg(measuredError),1)+'° (simulation truth)';
 $('matrixReadout').textContent='15×15 DIAGNOSTIC COVARIANCE · 1σ HDG '+nfmt(metrics.heading_sigma_deg,2)+
  '° · POSITION '+nfmt(metrics.position_sigma_m,2)+' m · RᵀR≈I';
 $('allocatorReadout').textContent=allocatedControl?
  'THRUST MATRIX ('+allocationPriority.toUpperCase()+') · Fₓ RESIDUAL '+nfmt(allocatedControl.forceError_N,1)+' N · Nₓ RESIDUAL '+
  nfmt(allocatedControl.momentError_Nm,1)+' N·m'+(allocatedControl.saturation?' · SATURATED':''):
  'THRUST MATRIX · READY';
 $('compassStatus').textContent=compass.ready?'PLOTLY 3D LIVE · TRUE-NORTH LOCK · '+(compass.linked?'CAMERA SYNC ON':'CAMERA SYNC OFF'):
  'CANVAS TRUE-HEADING BACKUP · '+(compass.linked?'CAMERA SYNC REQUESTED':'CAMERA UNLINKED');
}
function applyHeadingCommand(value,source='compass'){
 const h=bearing360(Number(value));if(!Number.isFinite(h))return false;
 if(challenge?.status==='running'){notice('Heading command unavailable during an active challenge.');return false;}
 if(s.failsafe||!s.c2||s.boat.batteryCutoff){notice('Heading command blocked by the active vessel safety state.');return false;}
 // Entering heading hold captures the current heading, so apply the requested
 // target AFTER the mode transition. Never mutate the true vessel pose.
 if(s.mode!=='heading')toggleMode('heading');
 if(s.mode!=='heading')return false;
 s.targetHeading=h;
 refreshButtons();updateCompassReadouts();twin.lastDraw=0;compass.lastDraw=0;
 notice('TRUE HDG '+Math.round(h)+'° commanded from '+source+'. USV turning through its 3/6-DOF dynamics.');
 return true;
}
function cameraFromPlotEvent(src,ev){
 // Plotly sends either the whole scene.camera or individual nested fields.
 const base=src==='vessel'?(twin.savedOrbit||cameraPreset()):(compass.savedCamera||compassCamera());
 const o=JSON.parse(JSON.stringify(base));let changed=false;
 if(ev['scene.camera']&&typeof ev['scene.camera']==='object')return JSON.parse(JSON.stringify(ev['scene.camera']));
 for(const key of ['eye','up','center'])for(const axis of ['x','y','z']){
  const k='scene.camera.'+key+'.'+axis,part=ev['scene.camera.'+key];
  if(typeof ev[k]==='number'&&Number.isFinite(ev[k])){(o[key]??={})[axis]=ev[k];changed=true;}
  else if(part&&typeof part[axis]==='number'&&Number.isFinite(part[axis])){(o[key]??={})[axis]=part[axis];changed=true;}
 }
 return changed?o:null;
}
function on3DCameraChanged(source,ev){
 if((source==='vessel'&&(twin.cameraSwitch||twin.pending))||
    (source==='compass'&&(compass.cameraSwitch||compass.pending)))return;
 const cam=cameraFromPlotEvent(source,ev);if(!cam)return;
 if(source==='vessel'){twin.savedOrbit=cam;}else compass.savedCamera=cam;
 if(!compass.linked){if(source==='vessel'){twin.camera='orbit';refreshTwinButtons();}return;}
 twin.camera='orbit';twin.savedOrbit=cam;compass.savedCamera=cam;
 refreshTwinButtons();twin.lastDraw=0;compass.lastDraw=0;
 if(source==='compass'&&twin.ready){twin.cameraSwitch=true;
  Promise.resolve(window.Plotly.relayout(twin.plot,{'scene.camera':cam})).catch(()=>{}).finally(()=>twin.cameraSwitch=false);}
 if(source==='vessel'&&compass.ready){compass.cameraSwitch=true;
  Promise.resolve(window.Plotly.relayout(compass.plot,{'scene.camera':cam})).catch(()=>{}).finally(()=>compass.cameraSwitch=false);}
 // Support orbiting the companion Plotly compass when Three.js is selected.
 if(source==='compass'&&three.ready){const e=cam.eye,azimuth=Math.atan2(e.y,e.x),elev=Math.atan2(e.z,Math.hypot(e.x,e.y));
  three.userOrbit={azimuth,elevation:elev,radius:11};}
 // The Canvas perspective viewer uses the same ENU orientation when linked.
 if(source==='compass'&&typeof preview!=='undefined') {
  const e=cam.eye;preview.azimuth=Math.atan2(e.y,e.x);preview.elevation=Math.atan2(e.z,Math.hypot(e.x,e.y));
  preview.last=0;}
}
function syncOrbitFromVesselRenderer(azimuth,elevation,radius){
 if(!compass.linked)return;
 const scale=clamp(radius/6,0.45,3.5);
 const cam={eye:{x:scale*Math.cos(elevation)*Math.cos(azimuth),
  y:scale*Math.cos(elevation)*Math.sin(azimuth),z:scale*Math.sin(elevation)},
  up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};
 twin.camera='orbit';twin.savedOrbit=cam;compass.savedCamera=cam;
 refreshTwinButtons();compass.lastDraw=0;
 if(compass.ready&&!compass.cameraSwitch){compass.cameraSwitch=true;
  Promise.resolve(window.Plotly.relayout(compass.plot,{'scene.camera':cam}))
  .catch(()=>{}).finally(()=>compass.cameraSwitch=false);}
}
function alignLinkedViews(){
 compass.linked=true;$('lock3D').classList.add('selected');$('lock3D').textContent='🔗 LOCK 3D VIEWS';
 $('cameraSyncStatus').textContent='CAMERA LINKED';
 const cam=compassCamera();compass.savedCamera=cam;twin.cameraSwitch=true;compass.cameraSwitch=true;
 const jobs=[];
 if(twin.ready)jobs.push(Promise.resolve(window.Plotly.relayout(twin.plot,{'scene.camera':cam})));
 if(compass.ready)jobs.push(Promise.resolve(window.Plotly.relayout(compass.plot,{'scene.camera':cam})));
 Promise.allSettled(jobs).finally(()=>{twin.cameraSwitch=false;compass.cameraSwitch=false;});
 compass.lastDraw=0;twin.lastDraw=0;updateCompassReadouts();
}
async function initializeCompass3D(){
 if(!compass.detailOpen||compass.ready||compass.loading||compass.plot.clientWidth<20)return;
 if(!window.Plotly||!webGLAvailable()){useCompassFallback();return;}
 compass.loading=true;
 try{
  await window.Plotly.newPlot(compass.plot,compass3DTraces(),compass3DLayout(),
   {responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['lasso2d','select2d']});
  compass.ready=true;compass.fallback.hidden=true;compass.plot.hidden=false;
  if(!compass.eventsBound){compass.eventsBound=true;
   compass.plot.on('plotly_relayout',ev=>on3DCameraChanged('compass',ev));
   compass.plot.on('plotly_click',ev=>{const pt=ev.points?.[0];
    if(pt&&pt.data?.name==='Click rim to command heading'&&Number.isFinite(Number(pt.customdata)))applyHeadingCommand(Number(pt.customdata),'compass rim');
   });
   compass.plot.addEventListener('webglcontextlost',()=>{compass.ready=false;useCompassFallback();},true);
  }
  if(compass.linked)alignLinkedViews();
 }catch(err){console.warn('Compass Plotly render failed',err);useCompassFallback();}
 finally{compass.loading=false;}
}
async function drawCompass3D(now=performance.now()){
 if(!compass.detailOpen||!compass.ready||compass.pending||twin.view==='2d'||activeWorkspace==='analysis'||document.hidden||compass.plot.clientWidth<20)return;
 const interval=window.matchMedia('(max-width:700px)').matches?320:185;
 if(now-compass.lastDraw<interval)return;
 compass.lastDraw=now;compass.pending=true;
 try{await window.Plotly.react(compass.plot,compass3DTraces(),compass3DLayout(),
  {responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['lasso2d','select2d']});}
 catch(e){console.warn('Compass 3D update failed',e);compass.ready=false;useCompassFallback();}
 finally{compass.pending=false;}
}
function useCompassFallback(){
 compass.ready=false;compass.plot.hidden=true;compass.fallback.hidden=false;updateCompassReadouts();drawCompassFallback();
}
function drawCompassFallback(){
 if(!compass.detailOpen||compass.fallback.hidden||twin.view==='2d')return;
 const cvs=compass.fallback,rect=cvs.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),w=Math.max(rect.width,200),h=Math.max(rect.height,240);
 if(cvs.width!==Math.round(w*dpr)||cvs.height!==Math.round(h*dpr)){cvs.width=Math.round(w*dpr);cvs.height=Math.round(h*dpr);}
 const c=cvs.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
 const cx=w/2,cy=h/2,radial=Math.min(w,h)*.39;
 c.fillStyle='#071a30';c.fillRect(0,0,w,h);c.beginPath();c.arc(cx,cy,radial,0,TAU);c.fillStyle='#102c53';c.fill();c.strokeStyle='#5ce8f7';c.lineWidth=4;c.stroke();
 for(let i=0;i<72;i++){const a=TAU*i/72-Math.PI/2,r0=radial*(i%6===0?.78:.86),r1=radial*.96;
  c.strokeStyle='#9fcde8';c.lineWidth=i%6===0?2:1;c.beginPath();c.moveTo(cx+Math.cos(a)*r0,cy+Math.sin(a)*r0);c.lineTo(cx+Math.cos(a)*r1,cy+Math.sin(a)*r1);c.stroke();}
 c.textAlign='center';c.textBaseline='middle';c.fillStyle='#f0faff';c.font='bold 16px system-ui';
 for(const [t,dx,dy] of [['N',0,-.65],['E',.65,0],['S',0,.65],['W',-.65,0]])c.fillText(t,cx+dx*radial,cy+dy*radial);
 const drawPointer=(h,col,width,r)=>{const a=rad(h-90);c.strokeStyle=col;c.lineWidth=width;c.beginPath();c.moveTo(cx,cy);c.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a));c.stroke();};
 // Compass uncertainty is a display of the diagnostic 15-state covariance,
 // not a guarantee of real-world sensor accuracy.
 const centerHeading=magneticSample.estimatedTrueHeading_deg,
  sigma=(s.est.insMetrics||LA.covarianceMetrics(s.est.insCov)).heading_sigma_deg,
  a0=rad(centerHeading-90-clamp(1.96*sigma,.1,65)),
  a1=rad(centerHeading-90+clamp(1.96*sigma,.1,65));
 c.beginPath();c.moveTo(cx,cy);c.arc(cx,cy,radial*.72,a0,a1,false);c.closePath();
 c.fillStyle='rgba(110,246,162,.12)';c.fill();
 c.beginPath();c.arc(cx,cy,radial*.72,a0,a1,false);
 c.strokeStyle='#6bea9a';c.lineWidth=2;c.stroke();
 drawPointer(s.targetHeading,'#ffd37d',4,radial*.64);
 drawPointer(centerHeading,'#83f2ac',3,radial*.58);
 drawPointer(trueBowDegrees(),'#40ebff',7,radial*.74);
 c.beginPath();c.arc(cx,cy,8,0,TAU);c.fillStyle='#fff';c.fill();
 c.fillStyle='#8fbfdd';c.font='11px system-ui';c.fillText('CANVAS BACKUP · TRUE NORTH',cx,h-18);
}
for(const id of ['compassHeadingInput','vesselHeadingInput']){
 $(id).addEventListener('input',ev=>applyHeadingCommand(ev.target.value,id==='vesselHeadingInput'?'USV display':'3D compass'));
}
// Sensor lab controls intentionally change ONLY simulated observations,
// never authoritative physical bow orientation or Plotly camera direction.
for(const [id,key] of [['hardIronInput','hardIron_uT'],['softIronInput','softIron_percent'],
 ['declinationInput','declination_deg']]){
 $(id).addEventListener('input',ev=>{
  magSettings[key]=Number(ev.target.value);
  $(id+'Value').textContent=(key==='softIron_percent'?String(magSettings[key])+'%':
   key==='hardIron_uT'?String(magSettings[key])+' µT':String(magSettings[key])+'°');
  sampleMagneticHeading();compass.lastDraw=0;updateCompassReadouts();
 });
}
$('magCalibrationInput').addEventListener('change',ev=>{
 magSettings.calibrated=ev.target.checked;
 sampleMagneticHeading();compass.lastDraw=0;updateCompassReadouts();
});
$('allocationPriorityInput').addEventListener('change',ev=>{
 allocationPriority=ev.target.value;
 updateCompassReadouts();
});
$('lock3D').addEventListener('click',()=>{compass.linked=!compass.linked;
 $('lock3D').classList.toggle('selected',compass.linked);
 $('lock3D').textContent=compass.linked?'🔗 LOCK 3D VIEWS':'🔓 VIEWS UNLINKED';
 $('cameraSyncStatus').textContent=compass.linked?'CAMERA LINKED':'INDEPENDENT CAMERAS';
 if(compass.linked)alignLinkedViews();updateCompassReadouts();});
$('align3D').addEventListener('click',alignLinkedViews);
$('toggleCompassDetail').addEventListener('click',()=>{
 compass.detailOpen=!compass.detailOpen;
 $('compassFrame').hidden=!compass.detailOpen;
 $('toggleCompassDetail').setAttribute('aria-expanded',String(compass.detailOpen));
 $('toggleCompassDetail').textContent=compass.detailOpen?'🧭 CLOSE COMPASS LAB':'🧭 OPEN COMPASS LAB';
 if(compass.detailOpen){
  requestAnimationFrame(()=>{
   if(compass.ready){window.Plotly.Plots.resize(compass.plot);compass.lastDraw=0;}
   else if(window.Plotly&&webGLAvailable())initializeCompass3D();
   else useCompassFallback();
   if(twin.ready)window.Plotly.Plots.resize(twin.plot);
   preview.last=0;
  });
 }else if(twin.ready){requestAnimationFrame(()=>window.Plotly.Plots.resize(twin.plot));}
});

function refreshTwinButtons(){
 $('views').classList.toggle('only-2d',twin.view==='2d');
 $('views').classList.toggle('only-3d',twin.view==='3d');
 document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===twin.view));
  document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('selected',b.dataset.camera===twin.camera));
  document.querySelectorAll('[data-renderer]').forEach(b=>b.classList.toggle('selected',b.dataset.renderer===twin.renderer));
}
function setTwinMessage(msg,show=true){$('twinMessage').textContent=msg;$('twinMessage').hidden=!show;}
function twinStatus(msg){$('twinStatusText').textContent=msg;}
function updateTwinInfo(){
 $('twinOrigin').textContent='LOCAL ENU · NORTH-FIXED COMPASS · '+twin.camera.toUpperCase()+' · '+(cfg.physicsMode==='6dof'?'6-DOF':'3-DOF');
 $('twinPose').textContent='E '+s.boat.x.toFixed(1)+' · N '+s.boat.y.toFixed(1)+' · Z '+(s.boat.z||0).toFixed(2)+' m · HDG '+Math.round(heading(s.boat.psi)).toString().padStart(3,'0')+'° · R/P '+deg(s.boat.roll||0).toFixed(1)+'°/'+deg(s.boat.pitch||0).toFixed(1)+'°';
}
async function drawTwin(now=performance.now()){
 if(!twin.ready||twin.pending||twin.view==='2d'||activeWorkspace==='analysis'||twin.renderer!=='plotly'||document.hidden||twin.plot.clientWidth<20)return;
 const small=window.matchMedia('(max-width: 700px)').matches;
 const interval=small?320:165; // about 3 Hz mobile, 6 Hz desktop
 if(now-twin.lastDraw<interval)return;
 twin.lastDraw=now;twin.pending=true;
 try{
  const data=makeTwinTraces(),layout=twinLayout();
  await window.Plotly.react(twin.plot,data,layout,{responsive:true,displaylogo:false,
   modeBarButtonsToRemove:['toImage','lasso2d','select2d'],scrollZoom:true});
  updateTwinInfo();
 }catch(e){console.warn('BLUEVECTOR 3D render error',e);twin.ready=false;useCanvasPreview('PLOTLY RENDERING FAILED');}
 finally{twin.pending=false;}
}
function setCamera(mode){
 twin.camera=mode;if(mode==='orbit')twin.savedOrbit=null;if(three.ready)three.userOrbit=null;preview.last=0;
 refreshTwinButtons();if(twin.ready){
  twin.cameraSwitch=true;
  window.Plotly.relayout(twin.plot,{'scene.camera':cameraPreset()}).catch(()=>{}).finally(()=>{twin.cameraSwitch=false;});
  twin.lastDraw=0;
 }
 if(compass.linked&&compass.ready){compass.cameraSwitch=true;
  window.Plotly.relayout(compass.plot,{'scene.camera':cameraPreset()}).catch(()=>{}).finally(()=>{compass.cameraSwitch=false;});
  compass.lastDraw=0;}
}
function setView(mode){
 twin.view=mode;refreshTwinButtons();
 if(mode!=='2d'){
  if(twin.renderer==='plotly'&&!twin.loading&&!twin.ready)loadPlotly();
  if(twin.renderer==='three'&&!three.loading&&!three.ready)loadThree();
  if(twin.renderer==='canvas')preview.last=0;
  if(twin.ready&&twin.renderer==='plotly'){requestAnimationFrame(()=>{window.Plotly.Plots.resize(twin.plot);twin.lastDraw=0;});}
  if(compass.detailOpen){
   if(compass.ready){requestAnimationFrame(()=>{window.Plotly.Plots.resize(compass.plot);compass.lastDraw=0;});}
   else if(window.Plotly&&webGLAvailable())initializeCompass3D();else useCompassFallback();
  }
 }
}
function webGLAvailable(){
 try{
  const test=document.createElement('canvas');
  const gl=test.getContext('webgl2')||test.getContext('webgl')||test.getContext('experimental-webgl');
  if(!gl)return false;
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  return true;
 }catch(e){return false;}
}
function loadPlotly(){
 if(twin.loading||twin.ready)return;
 if(!webGLAvailable()){
  useCanvasPreview('WEBGL UNAVAILABLE');return;
 }
 twin.loading=true;twin.renderer='plotly';twin.plot.hidden=false;three.view.hidden=true;preview.canvas.hidden=true;
 refreshTwinButtons();setTwinMessage('Loading interactive Plotly 3D USV…');twinStatus('Plotly.js initialization in progress');
 function fail(reason,e){
  console.warn('BLUEVECTOR Plotly:',reason,e||'');
  twin.loading=false;twin.ready=false;useCanvasPreview(reason);
 }
 async function initialize(){
  if(!window.Plotly||typeof window.Plotly.newPlot!=='function'){
   fail('PLOTLY.JS NOT LOADED');return;
  }
  try{
   await window.Plotly.newPlot(twin.plot,makeTwinTraces(),twinLayout(),
    {responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],scrollZoom:true});
   twin.ready=true;twin.loading=false;twin.lastDraw=performance.now();setTwinMessage('',false);
   twinStatus('PLOTLY 3D ACTIVE · USV + COMPASS IN ONE ENU SCENE · DRAG TO ORBIT');updateTwinInfo();
   if(compass.detailOpen)initializeCompass3D();
   twin.plot.on('plotly_click',e=>{
    if(!e.points||!e.points.length)return;
    const pt=e.points[0],nm=pt.data.name||'3D overlay';
   if(nm==='Integrated compass heading command'&&Number.isFinite(Number(pt.customdata))){
    applyHeadingCommand(Number(pt.customdata),'integrated compass rim');return;
   }
   $('twinComponent').textContent=modelInfo[nm]||nm;
   });
   twin.plot.on('plotly_relayout',ev=>on3DCameraChanged('vessel',ev));
   // A lost GPU context after initialization is not a missing vessel model.
   twin.plot.addEventListener('webglcontextlost',()=>{twin.ready=false;useCanvasPreview('WEBGL CONTEXT LOST');},true);
  }catch(e){fail('PLOTLY INITIALIZATION FAILED',e);}
 }
 if(window.Plotly){initialize();return;}
 const urls=['https://cdn.plot.ly/plotly-4.3.0.min.js',
  'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.3.0/plotly.min.js'];
 function tryCDN(i){
  if(i>=urls.length){fail('PLOTLY CDN UNREACHABLE');return;}
  const script=document.createElement('script');script.src=urls[i];script.async=true;
  let settled=false;
  const next=()=>{if(settled)return;settled=true;clearTimeout(timer);script.onload=null;script.onerror=null;tryCDN(i+1);};
  const timer=setTimeout(()=>{script.remove();next();},8000);
  script.onload=()=>{if(settled)return;settled=true;clearTimeout(timer);if(window.Plotly)initialize();else tryCDN(i+1);};
  script.onerror=next;
  document.head.append(script);
 }
 tryCDN(0);
}
// BLUEVECTOR 3D perspective fallback. This is Canvas 2D projection of the same
// triangulated hull and 6-DOF pose, not a WebGL Plotly renderer or second solver.
const preview={canvas:$('twinCanvas'),ctx:$('twinCanvas').getContext('2d'),last:0,
  azimuth:-2.37,elevation:.55,radius:10.5,drag:null};
function perspectiveBasis(){
 const b=s.boat,c=Math.cos(b.psi),sn=Math.sin(b.psi);
 let eye,up=[0,0,1];
 if(twin.camera==='overhead'){eye=[.01,.02,13];up=[0,1,0];}
 else if(twin.camera==='orbit'){
  eye=[preview.radius*Math.cos(preview.elevation)*Math.cos(preview.azimuth),
       preview.radius*Math.cos(preview.elevation)*Math.sin(preview.azimuth),
       preview.radius*Math.sin(preview.elevation)];
 }else eye=[-8.8*c+2.9*sn,-8.8*sn-2.9*c,5.3];
 const target=[0,0,.20],sub=(a,b)=>a.map((v,i)=>v-b[i]);
 const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
 const norm=a=>{const l=Math.hypot(...a)||1;return a.map(v=>v/l);};
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const f=norm(sub(target,eye)),r=norm(cross(f,up)),u=norm(cross(r,f));
 return {eye,f,r,u,dot,cross,norm,sub};
}
function previewProject(p,basis,w,h){
 const d=basis.sub(p,basis.eye),depth=basis.dot(d,basis.f);
 if(depth<.3)return null;
 // Favor showing the full compass rose over an excessively zoomed hull.
 const focal=Math.min(w*.98,h*1.42);
 return [w*.5+focal*basis.dot(d,basis.r)/depth,
         h*.51-focal*basis.dot(d,basis.u)/depth,depth];
}
function previewRGB(hex,shade=1){
 const v=parseInt(hex.replace('#',''),16);
 const r=Math.min(255,Math.max(0,Math.round((v>>16)*shade))),
       g=Math.min(255,Math.max(0,Math.round(((v>>8)&255)*shade))),
       b=Math.min(255,Math.max(0,Math.round((v&255)*shade)));
 return `rgb(${r},${g},${b})`;
}
function renderPreview(now){
 if(twin.renderer!=='canvas'||twin.view==='2d'||activeWorkspace==='analysis'||document.hidden)return;
 const mobile=window.matchMedia('(max-width:700px)').matches;
 if(now-preview.last<(mobile?80:55))return;preview.last=now;
 const canvas=preview.canvas,ctx=preview.ctx,rect=canvas.getBoundingClientRect();
 const w=Math.max(1,rect.width),h=Math.max(1,rect.height);
 const dpr=Math.min(devicePixelRatio||1,2);
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
 }
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 let grad=ctx.createLinearGradient(0,0,0,h);grad.addColorStop(0,'#071c37');grad.addColorStop(.55,'#0c3958');grad.addColorStop(1,'#041629');
 ctx.fillStyle=grad;ctx.fillRect(0,0,w,h);
 const B=perspectiveBasis(),pr=p=>previewProject(p,B,w,h);
 const drawLine=(p,q,color,width=1,dashed=false)=>{
  const a=pr(p),b=pr(q);if(!a||!b)return;
  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;if(dashed)ctx.setLineDash([5,5]);
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.restore();
 };
 // Projected local ENU water grid. Water is illustrative; never integrated as hydrodynamic state.
 const surf=(x,y)=>[x,y,cfg.physicsMode==='6dof'?window.Bluevector6DOF.waveAt(s.boat.x+x,s.boat.y+y,s.t,
  {height:cfg.waveHeight,period:cfg.wavePeriod,bearing:cfg.waveBearing}).z :-.24];
 ctx.save();
 for(let v=-12;v<=12;v+=2){drawLine(surf(-12,v),surf(12,v),'#17658b90',.7);drawLine(surf(v,-12),surf(v,12),'#1b6c8a80',.7);}
 ctx.restore();
 // Compass rose is world-fixed; the visible boat always occupies its center.
 // Draw behind hull geometry so it cannot hide the physical vessel silhouette.
 const compassZ=integratedCompass.ringHeight,R=integratedCompass.ringRadius;
 if(integratedCompass.visible){
  for(const radius of [R,R-.38]){
   let prev=null;
   for(let k=0;k<=96;k++){const a=TAU*k/96,p=[radius*Math.cos(a),radius*Math.sin(a),compassZ];
    if(prev)drawLine(prev,p,radius===R?'#62e9fa':'#478db8',radius===R?2.5:1.4);
    prev=p;
   }
  }
  for(let h=0;h<360;h+=10){const a=rad(90-h),r0=h%30===0?R-.35:R-.2;
   drawLine([r0*Math.cos(a),r0*Math.sin(a),compassZ],[(R-.05)*Math.cos(a),(R-.05)*Math.sin(a),compassZ],
    '#bcecff',h%30===0?1.8:.8);
  }
  ctx.save();ctx.font='bold 16px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
  for(const [name,east,north] of [['N',0,R+.48],['E',R+.48,0],['S',0,-R-.48],['W',-R-.48,0]]){
   const pt=pr([east,north,compassZ+.25]);if(pt){
    ctx.lineWidth=3;ctx.strokeStyle='#09243a';ctx.strokeText(name,pt[0],pt[1]);
    ctx.fillStyle='#d8f7ff';ctx.fillText(name,pt[0],pt[1]);
   }
  }
  ctx.restore();
 }
 const light=B.norm([-1,-.7,2.7]);const faces=[];
 // Every body vertex uses the same ENU transformation as Plotly and the 2D chart.
 for(const [name,g] of Object.entries(vesselGeometry.groups)){
  for(let k=0;k<g.verts.length;k+=3){
   const a=bodyToENU(g.verts[k]),b=bodyToENU(g.verts[k+1]),c=bodyToENU(g.verts[k+2]);
   const pa=pr(a),pb=pr(b),pc=pr(c);
   if(!pa||!pb||!pc)continue;
   const ab=B.sub(b,a),ac=B.sub(c,a),n=B.norm(B.cross(ab,ac));
   const lightFactor=.61+.36*Math.abs(B.dot(n,light));
   faces.push({p:[pa,pb,pc],depth:(pa[2]+pb[2]+pc[2])/3,
    color:previewRGB(meshColors[name]||'#447ab1',lightFactor),name});
  }
 }
 faces.sort((a,b)=>b.depth-a.depth);
 for(const face of faces){
  const v=face.p;ctx.fillStyle=face.color;ctx.beginPath();ctx.moveTo(v[0][0],v[0][1]);
  ctx.lineTo(v[1][0],v[1][1]);ctx.lineTo(v[2][0],v[2][1]);ctx.closePath();ctx.fill();
  if(twin.layers.wireframe){ctx.strokeStyle='#84d9f070';ctx.lineWidth=.35;ctx.stroke();}
 }
 for(const edge of vesselGeometry.edges){for(let k=0;k<edge.length-1;k++)drawLine(bodyToENU(edge[k]),bodyToENU(edge[k+1]),'#60c7ee',1.1);}
 if(twin.layers.thrust){
  for(const [side,force,color] of [[1,s.boat.port,'#59dbff'],[-1,s.boat.star,'#7692ff']]){
   const a=[-1.68,side*hull.arm,hull.sixDof.thrusterZ],b=[a[0]+1.1*force/hull.maxThrust,a[1],a[2]];
   drawLine(bodyToENU(a),bodyToENU(b),color,3.2);
  }
 }
 if(twin.layers.axes){
  for(const [end,color] of [[[2.9,0,.36],'#f5a4ad'],[[0,2.2,.36],'#91f2bd'],[[0,0,2.3],'#92b8fa']])drawLine(bodyToENU([0,0,.36]),bodyToENU(end),color,1.8);
 }
 if(twin.layers.route){
  const wp=(challenge&&challenge.status==='running')?challengeGoal(challenge):(s.mode==='auto'?s.navRoute[s.navIndex]:s.waypoints[Math.min(s.wpIndex,s.waypoints.length-1)]);
  if(wp){const dx=wp.x-s.boat.x,dy=wp.y-s.boat.y,L=Math.hypot(dx,dy)||1;
   drawLine([0,0,.3],[6.5*dx/L,6.5*dy/L,.30],'#ffd68b',2,true);}
 }
 if(twin.layers.environment){
  drawLine([-4,-5,.5],[-4+current.x*7,-5+current.y*7,.5],'#67ffe6',2.5);
  drawLine([-4,5,.5],[-4+wind.x*.35,5+wind.y*.35,.5],'#ffd87d',2.5);
 }
 if(twin.layers.wake&&Math.hypot(s.boat.u,s.boat.v)>.25){for(const side of [-1,1]){
  let prev=null;for(let k=0;k<=18;k++){const t=k/18,cur=bodyToENU([-1.8-5.3*t,side*(.55+1.9*t),.03]);
   if(prev)drawLine(prev,cur,'#86e8ff99',1.7);prev=cur;}}
 }
 if(integratedCompass.visible){
  const R=integratedCompass.ringRadius,z=integratedCompass.ringHeight+.25;
  const bow=vesselBowENU(),hdg=rad(90-s.targetHeading),mag=rad(90-magneticSample.estimatedTrueHeading_deg);
  const cmd=[Math.cos(hdg),Math.sin(hdg)],estimate=[Math.cos(mag),Math.sin(mag)];
  for(const [v,r0,r1,col,width] of [[bow,2.22,R-.48,'#46f0ff',4.6],
    [estimate,2.54,R-.83,'#80efac',2.6],[cmd,2.37,R-.65,'#ffd987',3.1]]){
   drawLine([r0*v[0],r0*v[1],z],[r1*v[0],r1*v[1],z],col,width);
   const tip=pr([r1*v[0],r1*v[1],z]);if(tip){ctx.fillStyle=col;ctx.beginPath();ctx.arc(tip[0],tip[1],3.2,0,TAU);ctx.fill();}
  }
 }
 ctx.fillStyle='#b9eaff';ctx.font='bold 11px system-ui';ctx.fillText('3D PERSPECTIVE · CANVAS FALLBACK',12,22);
 ctx.font='10px system-ui';ctx.fillStyle='#8dbbd2';ctx.fillText('Same 6-DOF pose · Drag to orbit · No WebGL',12,38);
 updateTwinInfo();
}
preview.canvas.addEventListener('pointerdown',e=>{
 preview.drag={x:e.clientX,y:e.clientY};preview.canvas.setPointerCapture(e.pointerId);
 if(twin.camera!=='orbit'){
  const c=Math.cos(s.boat.psi),sn=Math.sin(s.boat.psi);
  preview.azimuth=Math.atan2(-8*sn-3*c,-8*c+3*sn);preview.elevation=.5;
  twin.camera='orbit';refreshTwinButtons();
 }
});
preview.canvas.addEventListener('pointermove',e=>{
 if(!preview.drag)return;
 preview.azimuth+=(e.clientX-preview.drag.x)*.008;
 preview.elevation=clamp(preview.elevation+(e.clientY-preview.drag.y)*.007,-.15,1.46);
 preview.drag={x:e.clientX,y:e.clientY};preview.last=0;syncOrbitFromVesselRenderer(preview.azimuth,preview.elevation,preview.radius);
});
preview.canvas.addEventListener('pointerup',()=>preview.drag=null);
preview.canvas.addEventListener('pointercancel',()=>preview.drag=null);
preview.canvas.addEventListener('wheel',e=>{e.preventDefault();
 preview.radius=clamp(preview.radius*Math.exp(Math.sign(e.deltaY)*.09),4,24);preview.last=0;syncOrbitFromVesselRenderer(preview.azimuth,preview.elevation,preview.radius);
},{passive:false});
function useCanvasPreview(reason){
 twin.renderer='canvas';twin.plot.hidden=true;three.view.hidden=true;preview.canvas.hidden=false;
 refreshTwinButtons();setTwinMessage('',false);
 $('twinStatusText').textContent='CANVAS PERSPECTIVE · '+reason+' · 3D geometry/pose remains synchronized';
 preview.last=0;renderPreview(performance.now());if(!compass.ready)useCompassFallback();
}

// Optional Three.js view shares the same triangulated hull and the same 60 Hz truth state.
// In 6-DOF mode vessel attitude is physical; in 3-DOF mode the water is decorative.
// Both modes retain an illustrative water material and wake (not CFD).
const three={ready:false,loading:false,failed:false,last:0,view:$('threeView'),scene:null,
 camera:null,renderer:null,vessel:null,water:null,THREE:null,forceArrows:[],userOrbit:null};
function threeWebGLAvailable(){return webGLAvailable();}
async function loadThree(){
 if(three.ready||three.loading)return;
 if(!threeWebGLAvailable()){useCanvasPreview('WEBGL UNAVAILABLE');return;}
 three.loading=true;setTwinMessage('Loading optional cinematic renderer…');
 try{
  const THREE=await import('https://cdn.jsdelivr.net/npm/three@0.181.2/build/three.module.js');three.THREE=THREE;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#06162c');scene.fog=new THREE.FogExp2('#061c38',.045);
  const camera=new THREE.PerspectiveCamera(42,1,.1,100);camera.up.set(0,0,1);
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.setClearColor('#06162c',1);three.view.appendChild(renderer.domElement);
  three.view.insertAdjacentHTML('beforeend','<span class="overlay-label">SIMULATED 6-AXIS POSE WHEN ENABLED · ILLUSTRATIVE OCEAN · LOCAL ENU</span>');
  scene.add(new THREE.HemisphereLight('#b3e6ff','#062040',2.1));
  const sun=new THREE.DirectionalLight('#d4efff',2.4);sun.position.set(-5,-5,11);scene.add(sun);
  const fill=new THREE.DirectionalLight('#278fff',.9);fill.position.set(6,4,5);scene.add(fill);
  const group=new THREE.Group();scene.add(group);
  const toColor={'Hull shell':'#13385e','Deck / roof':'#d9e9ff','Royal-blue sheer stripe':'#4169e1',
   'Cabin glazing':'#0b79b9','Sensor hardware':'#d9f2ff','Twin thruster housings':'#124576'};
  for(const [name,g] of Object.entries(vesselGeometry.groups)){
   const vertices=new Float32Array(g.verts.flat());const geo=new THREE.BufferGeometry();
   geo.setAttribute('position',new THREE.BufferAttribute(vertices,3));geo.computeVertexNormals();
   const mat=new THREE.MeshStandardMaterial({color:toColor[name]||'#ffffff',metalness:.27,roughness:.34,side:THREE.DoubleSide});
   const mesh=new THREE.Mesh(geo,mat);mesh.name=name;group.add(mesh);
  }
  const lineMat=new THREE.LineBasicMaterial({color:'#5bcaff',transparent:true,opacity:.65});
  for(const path of vesselGeometry.edges){const g=new THREE.BufferGeometry().setFromPoints(path.map(p=>new THREE.Vector3(...p)));group.add(new THREE.Line(g,lineMat));}
  // Animated ocean shading is illustrative. No CFD / wave loading is generated here.
  const mobile=window.matchMedia('(max-width:700px)').matches;
  const waterGeom=new THREE.PlaneGeometry(44,44,mobile?22:40,mobile?22:40);
  const waterMat=new THREE.MeshStandardMaterial({color:'#075d91',metalness:.28,roughness:.22,transparent:true,opacity:.86,side:THREE.DoubleSide,flatShading:false});
  const water=new THREE.Mesh(waterGeom,waterMat);water.position.z=0;scene.add(water);
  const grid=new THREE.GridHelper(32,16,'#13a3c6','#176a91');grid.rotation.x=Math.PI/2;grid.position.z=-.39;scene.add(grid);
  const dirAxes=new THREE.AxesHelper(2);dirAxes.position.set(-6,-6,0);scene.add(dirAxes);
  // Earth-fixed compass: the vessel moves within the ring, never the reverse.
  const cr=integratedCompass.ringRadius,cz=integratedCompass.ringHeight;
  const rose=new THREE.Group();scene.add(rose);
  const ringGeo=new THREE.RingGeometry(cr-.36,cr,96);
  rose.add(new THREE.Mesh(ringGeo,new THREE.MeshBasicMaterial({color:'#276dab',transparent:true,opacity:.37,side:THREE.DoubleSide,depthWrite:false})));
  const circlePoints=Array.from({length:97},(_,i)=>new THREE.Vector3(cr*Math.cos(TAU*i/96),cr*Math.sin(TAU*i/96),.02));
  rose.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePoints),new THREE.LineBasicMaterial({color:'#52e9f9'})));
  for(let h=0;h<360;h+=10){const a=rad(90-h),r0=h%30===0?cr-.34:cr-.2;
   rose.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(r0*Math.cos(a),r0*Math.sin(a),.08),
    new THREE.Vector3((cr-.04)*Math.cos(a),(cr-.04)*Math.sin(a),.08)]),
    new THREE.LineBasicMaterial({color:'#bfeaff'})));
  }
  // Cardinal labels use built-in browser canvas textures, not additional files.
  for(const [letter,x,y] of [['N',0,cr+.52],['E',cr+.52,0],['S',0,-cr-.52],['W',-cr-.52,0]]){
   const label=document.createElement('canvas');label.width=64;label.height=64;
   const cx=label.getContext('2d');cx.font='bold 47px Arial';cx.textAlign='center';cx.textBaseline='middle';
   cx.fillStyle='#e8faff';cx.fillText(letter,32,33);
   const texture=new THREE.CanvasTexture(label),sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false}));
   sprite.position.set(x,y,.25);sprite.scale.set(.65,.65,1);rose.add(sprite);
  }
  rose.position.z=cz;
  const arrowColors=['#3ff4ff','#86f5b0','#ffd37d'];
  const headingArrows=arrowColors.map(color=>{
   const a=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),new THREE.Vector3(0,2.2,.6),1.3,color,.29,.19);
   scene.add(a);return a;
  });
  const arrowP=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(-1.68,hull.arm,hull.sixDof.thrusterZ),0,'#4cd2ff',.25,.12);
  const arrowS=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(-1.68,-hull.arm,hull.sixDof.thrusterZ),0,'#4cd2ff',.25,.12);
  group.add(arrowP,arrowS);
  three.forceArrows=[arrowP,arrowS];
  three.scene=scene;three.camera=camera;three.renderer=renderer;three.vessel=group;three.water=water;
  three.compassRose=rose;three.compassArrows=headingArrows;
  let pointer=null, azimuth=-2.4,elevation=.40,radius=11;
  renderer.domElement.addEventListener('pointerdown',e=>{if(three.userOrbit)({azimuth,elevation,radius}=three.userOrbit);pointer={x:e.clientX,y:e.clientY};renderer.domElement.setPointerCapture(e.pointerId);});
  renderer.domElement.addEventListener('pointermove',e=>{if(!pointer)return;
    azimuth+=(e.clientX-pointer.x)*.008;elevation=clamp(elevation+(e.clientY-pointer.y)*.006,.10,1.4);
    pointer={x:e.clientX,y:e.clientY};three.userOrbit={azimuth,elevation,radius};twin.camera='orbit';refreshTwinButtons();syncOrbitFromVesselRenderer(azimuth,elevation,radius);});
  renderer.domElement.addEventListener('pointerup',()=>pointer=null);
  renderer.domElement.addEventListener('pointercancel',()=>pointer=null);
  renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();radius=clamp(radius*(1+Math.sign(e.deltaY)*.07),5,25);
    three.userOrbit={azimuth,elevation,radius};twin.camera='orbit';refreshTwinButtons();syncOrbitFromVesselRenderer(azimuth,elevation,radius);},{passive:false});
  three.ready=true;setTwinMessage('',false);twinStatus('THREE.JS CINEMATIC ACTIVE · LIVE SIMULATION POSE');renderThree(performance.now());
 }catch(e){three.failed=true;console.warn('Three.js load failed',e);useCanvasPreview('THREE.JS UNAVAILABLE');}
 finally{three.loading=false;}
}
function renderThree(now){
 if(!three.ready||three.renderer===null||twin.renderer!=='three'||twin.view==='2d'||activeWorkspace==='analysis'||document.hidden)return;
 const isMobile=window.matchMedia('(max-width:700px)').matches;
 if(now-three.last<(isMobile?45:30))return;three.last=now;
 const T=three.THREE,b=s.boat,model=three.vessel;
 if(cfg.physicsMode==='6dof'){
  // Three.js world axes equal ENU: x East, y North, z Up.
  // Apply the exact unit quaternion used by the dynamics solver.
  model.quaternion.set(b.attitude[1],b.attitude[2],b.attitude[3],b.attitude[0]);
  model.position.z=b.z;
 }else{model.rotation.set(0,0,b.psi);model.position.z=0;}
 if(three.compassRose){three.compassRose.visible=integratedCompass.visible;}
 if(three.compassArrows){
  const bow=vesselBowENU(),m=rad(90-magneticSample.estimatedTrueHeading_deg),h=rad(90-s.targetHeading),
    dirs=[bow,[Math.cos(m),Math.sin(m)],[Math.cos(h),Math.sin(h)]];
  dirs.forEach((v,i)=>{
   const arrow=three.compassArrows[i],r0=[2.22,2.54,2.37][i],len=[1.55,.87,1.13][i];
   arrow.position.set(r0*v[0],r0*v[1],integratedCompass.ringHeight+.25+i*.05);
   arrow.setDirection(new T.Vector3(v[0],v[1],0));arrow.setLength(len,.29,.18);arrow.visible=integratedCompass.visible;
  });
 }
 const c=Math.cos(b.psi),sn=Math.sin(b.psi);
 const target=new T.Vector3(0,0,.4);
 if(twin.camera==='overhead'){three.camera.position.set(0,0,14);three.camera.up.set(0,1,0);}
 else if(twin.camera==='orbit'){
  const o=three.userOrbit||{azimuth:-2.4,elevation:.40,radius:11};three.camera.position.set(o.radius*Math.cos(o.elevation)*Math.cos(o.azimuth),o.radius*Math.cos(o.elevation)*Math.sin(o.azimuth),o.radius*Math.sin(o.elevation));
 }else{
  three.camera.position.set(-8*c+3*sn,-8*sn-3*c,5.2);
 }
 if(twin.camera!=='overhead')three.camera.up.set(0,0,1);three.camera.lookAt(target);
 const geometry=three.water.geometry,positions=geometry.attributes.position;
 for(let k=0;k<positions.count;k++){
  const x=positions.getX(k),y=positions.getY(k);
  if(cfg.physicsMode==='6dof')positions.setZ(k,window.Bluevector6DOF.waveAt(b.x+x,b.y+y,s.t,
    {height:cfg.waveHeight,period:cfg.wavePeriod,bearing:cfg.waveBearing}).z);
  else positions.setZ(k,.05*Math.sin(.75*x+.38*y+now*.0015)+.03*Math.cos(.6*y-.2*x-now*.001));
 }positions.needsUpdate=true;geometry.computeVertexNormals();
 const [p,sr]=three.forceArrows;
 p.setLength(clamp(Math.abs(b.port)/hull.maxThrust*1.4,.04,1.4),.22,.13);
 sr.setLength(clamp(Math.abs(b.star)/hull.maxThrust*1.4,.04,1.4),.22,.13);
 p.setDirection(new T.Vector3(b.port>=0?1:-1,0,0));sr.setDirection(new T.Vector3(b.star>=0?1:-1,0,0));
 p.visible=twin.layers.thrust;sr.visible=twin.layers.thrust;
 const rect=three.view.getBoundingClientRect();const w=Math.max(1,rect.width),h=Math.max(1,rect.height);
 if(three.renderer.domElement.width!==Math.floor(w*Math.min(devicePixelRatio||1,2))||three.renderer.domElement.height!==Math.floor(h*Math.min(devicePixelRatio||1,2))){
   three.renderer.setSize(w,h,false);three.camera.aspect=w/h;three.camera.updateProjectionMatrix();
 }
 three.renderer.render(three.scene,three.camera);updateTwinInfo();
}
function setTwinRenderer(type){
 twin.renderer=type;
 twin.plot.hidden=type!=='plotly';three.view.hidden=type!=='three';preview.canvas.hidden=type!=='canvas';
 refreshTwinButtons();
 if(type==='plotly'){
  if(!twin.ready&&!twin.loading)loadPlotly();
  else if(twin.ready){setTwinMessage('',false);twinStatus('PLOTLY 3D ACTIVE · LIVE SIMULATION POSE');
    requestAnimationFrame(()=>{window.Plotly.Plots.resize(twin.plot);twin.lastDraw=0;});}
 }else if(type==='three'){
  if(three.ready){setTwinMessage('',false);twinStatus('THREE.JS CINEMATIC ACTIVE · LIVE SIMULATION POSE');renderThree(performance.now());}
  else if(!three.loading)loadThree();
 }else useCanvasPreview('USER SELECTED');
}
document.querySelectorAll('[data-renderer]').forEach(btn=>btn.addEventListener('click',()=>setTwinRenderer(btn.dataset.renderer)));
document.querySelectorAll('[data-camera]').forEach(btn=>btn.addEventListener('click',()=>setCamera(btn.dataset.camera)));
document.querySelectorAll('[data-view]').forEach(btn=>btn.addEventListener('click',()=>setView(btn.dataset.view)));
document.querySelectorAll('[data-layer]').forEach(el=>el.addEventListener('change',()=>{twin.layers[el.dataset.layer]=el.checked;twin.lastDraw=0;}));
window.addEventListener('resize',()=>{if(twin.ready&&twin.view!=='2d'&&twin.renderer==='plotly')window.Plotly.Plots.resize(twin.plot);if(compass.ready&&twin.view!=='2d')window.Plotly.Plots.resize(compass.plot);preview.last=0;});
$('retryPlotly').addEventListener('click',()=>{twin.ready=false;twin.loading=false;setTwinRenderer('plotly');});
refreshTwinButtons();loadPlotly();
// ======================= END BLUEVECTOR 3D PLOTLY TWIN =======================

// Fixed 60 Hz integration: graphical frames never determine model integration dt.
let last=performance.now(),accumulator=0;
function frame(now){const elapsed=Math.min(0.2,Math.max(0,(now-last)/1000));last=now;
 if(!paused){accumulator+=elapsed*simRate;let steps=0;while(accumulator>=DT&&steps<12){update(DT);accumulator-=DT;steps++;}if(steps===12)accumulator=0;}
 drawMap();display();drawTwin(now);drawCompass3D(now);drawCompassFallback();updateCompassReadouts();renderThree(now);renderPreview(now);bluevectorPublish();requestAnimationFrame(frame);
}
// --- BLUEVECTOR portable browser integration contract (v1) ---
// The simulation remains authoritative. External UIs may read snapshots or
// dispatch validated control intents without importing Streamlit or Plotly.
// Units are SI; ENU is x-East/y-North/z-Up; body axes x-forward/y-port/z-up.
const _bvObservers = new Set();
let _bvLastPublish = -Infinity;
function bluevectorSnapshot(){
 const b=s.boat, e=s.est;
 return {
  schema:'bluevector.state.v1', contractVersion:'1.0',
  time_s:s.t, physicsMode:cfg.physicsMode,
  pose:{frame:'ENU',east_m:b.x,north_m:b.y,up_m:b.z||0,
   quaternion_body_to_enu:(b.attitude||window.Bluevector6DOF.fromEuler(b.roll||0,b.pitch||0,b.psi)).slice(),
   yaw_rad:b.psi,roll_rad:b.roll||0,pitch_rad:b.pitch||0},
  velocity:{frame:'BODY_WATER_RELATIVE',surge_mps:b.u,sway_mps:b.v,heave_mps:b.w||0,
   roll_rate_radps:b.p||0,pitch_rate_radps:b.q||0,yaw_rate_radps:b.r},
  propulsion:{port_N:b.port,starboard_N:b.star,power_W:b.power},
  energy:{used_Wh:b.energy,battery_capacity_Wh:hull.batteryWh},
  compass:{true_heading_deg:trueBowDegrees(),command_heading_deg:s.targetHeading,
   magnetometer_true_heading_deg:magneticSample.estimatedTrueHeading_deg,
   sensor_heading_error_deg:((magneticSample.estimatedTrueHeading_deg-trueBowDegrees()+540)%360)-180,
   sensor_calibrated:magSettings.calibrated,declination_deg:magSettings.declination_deg,
   hard_iron_uT:magSettings.hardIron_uT,soft_iron_percent:magSettings.softIron_percent,
   heading_sigma_deg:s.est.insMetrics?.heading_sigma_deg??null,
   true_north_frame:'ENU',camera_linked:compass.linked},
  actuatorAllocation:allocatedControl?{...allocatedControl,priority:allocationPriority}:null,
  navigation:{east_m:e.x,north_m:e.y,heading_rad:e.psi,
   sigma_m:e.sigma,diagnostic_15state_sigma_m:e.insMetrics?.position_sigma_m??null,
   gnss_available:s.gnss,c2_available:s.c2},
  mission:{operating_mode:s.mode,waypoint_index:s.wpIndex,
   waypoint_count:s.waypoints.length,finished:s.finished,
   failsafe:s.failsafe,collisions:s.collisions,
   challenge_type:challenge?.type||null,challenge_status:challenge?.status||null}
 };
}
function bluevectorCommand(action){
 switch(action){
  case 'pause':paused=true;return true;
  case 'play':paused=false;return true;
  case 'reset':reset();return true;
  case 'align-3d-views':alignLinkedViews();return true;
  case 'manual':case 'heading':case 'auto':
   if(challenge?.status==='running'||(action==='auto'&&!s.waypoints.length))return false;
   toggleMode(action);return s.mode===action;
  default:return false;
 }
}
window.BluevectorApp = Object.freeze({
 contractVersion:'1.0',
 capabilities:Object.freeze(['snapshot','subscribe','command','heading-command','linked-3d-compass',
  'matrix-transform','bounded-thrust-allocation','magnetometer-calibration','15state-diagnostic-covariance','3dof','6dof','mission-challenges']),
 commandHeading(degrees){return applyHeadingCommand(degrees,'portable API');},
 snapshot:bluevectorSnapshot,
 command:bluevectorCommand,
 subscribe(handler){
  if(typeof handler!=='function')throw new TypeError('subscriber must be a function');
  _bvObservers.add(handler);
  return ()=>_bvObservers.delete(handler);
 }
});
function bluevectorPublish(){
 if(s.t-_bvLastPublish<0.2)return;
 _bvLastPublish=s.t;
 if(!_bvObservers.size)return;
 const snapshot=bluevectorSnapshot();
 for(const fn of _bvObservers){try{fn(snapshot);}catch(err){console.warn('BLUEVECTOR subscriber error',err);}}
}
// --- End portable integration contract ---

buildHelm();logEvent('Ready · manual mode · 4-waypoint demo route');drawMap();display();requestAnimationFrame(frame);
})();
'''

BROWSER_PAGE_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<style>__BLUEVECTOR_CSS__</style>
</head>
<body>
<div class="shell" id="shell" data-workspace="twin">
<header>
 <div class="brand"><div class="logo">BV</div><div><h1>BLUEVECTOR</h1><div class="sub">Autonomous USV Simulation &amp; Digital Twin</div></div></div>
 <div class="header-right"><span class="pill"><i class="statusdot" id="statusDot"></i><span id="statusText">READY</span></span><span class="pill" id="timeText">T+ 00:00</span><span class="pill">SIM ONLY</span></div>
</header>
<div class="workspace-bar" role="group" aria-label="BLUEVECTOR workspaces">
 <button class="action" data-workspace="helm">◉ Helm</button>
 <button class="action" data-workspace="mission">⌖ Mission</button>
 <button class="action selected" data-workspace="twin">⬡ 3D Digital Twin</button>
 <button class="action" data-workspace="analysis">▥ Analytics</button>
 <button class="action" data-workspace="challenge">★ Mission Challenge</button>
 <label class="speed-wrap">SIM RATE <select id="timeScale" aria-label="Simulation time scale"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
 <span class="badge" id="modelTag">COUPLED 3-DOF · EKF</span>
</div>
<div class="layout">
 <div class="main-col">
   <div class="view-bar" role="group" aria-label="Simulation visualization mode">
     <span class="view-title">VISUALIZATION</span>
     <div class="view-buttons"><button class="action" data-view="2d">2D Chart</button><button class="action" data-view="both">2D + 3D</button><button class="action selected" data-view="3d">3D Twin</button></div>
     <span class="view-note">One physics state · ENU axes · 6-DOF attitude synchronized when selected</span>
   </div>
   <div class="views only-3d" id="views">
   <div class="map-frame" id="mapFrame">
    <canvas id="chart" width="1200" height="675" aria-label="Interactive USV nautical chart; click to place mission waypoints while plan mode is enabled"></canvas>
    <div class="map-label"><strong>SECTOR BLUEWATER / LOCAL ENU</strong><span>180 × 101.25 m · generated demonstration chart</span></div>
    <div class="map-readout"><div><strong id="mapSpeed">0.0</strong><span>SOG m/s</span></div><div><strong id="mapHeading">090°</strong><span>Heading</span></div><div><strong id="mapWP">0 / 4</strong><span>Waypoint</span></div></div>
   </div>
   <div class="twin-pair" id="twinPair">
   <section class="twin-frame" id="twinFrame" aria-label="Interactive 3D Plotly vessel digital twin">
     <div class="twin-header"><strong>🛥️ 3D USV + TRUE-NORTH COMPASS</strong><span id="twinOrigin">LOCAL ENU · VESSEL CENTERED</span></div>
     <div class="camera-bar" role="group" aria-label="3D camera presets">
       <button class="action selected" data-camera="chase">◉ Chase</button>
       <button class="action" data-camera="overhead">⊕ Overhead</button>
       <button class="action" data-camera="orbit">↻ Free orbit</button>
       <span class="tool-label" style="margin-left:auto">RENDERER</span>
       <button class="action selected" data-renderer="plotly">Engineering Plotly</button>
       <button class="action" data-renderer="three">Cinematic Three.js</button>
       <button class="action" data-renderer="canvas">Canvas Preview</button>
     </div>
     <div id="twinPlot" aria-label="Combined Plotly 3D USV with surrounding, fixed-north compass rose. Drag to orbit the combined scene and click compass rim markers to command heading."></div>
     <div id="threeView" hidden aria-label="Cinematic Three.js USV model; drag to orbit, scroll to zoom"></div>
     <canvas id="twinCanvas" hidden aria-label="Non-WebGL perspective preview with the USV within the surrounding fixed-north compass rose; drag to orbit"></canvas>
     <div class="twin-message" id="twinMessage">Loading interactive Plotly 3D engine…</div>
     <div class="twin-status" id="twinStatus"><span id="twinStatusText" role="status">Initializing 3D vessel...</span><button id="retryPlotly" type="button">↻ Retry Plotly</button></div>
     <div class="twin-info"><span id="twinPose">POSITION E 21.0 · N 48.0 m</span><span id="twinComponent">Tap hull or components to inspect</span></div>
     <div class="combined-compass-strip" aria-label="Combined compass legend and readouts">
       <strong id="combinedHeadingReadout">TRUE 090° · MAG 090° · CMD 090°</strong>
       <span class="legend legend-cyan">● TRUE BOW</span>
       <span class="legend legend-green">● MAGNETIC EST.</span>
       <span class="legend legend-amber">● TARGET</span>
       <button type="button" id="toggleCompassDetail" aria-controls="compassFrame" aria-expanded="false">🧭 OPEN COMPASS LAB</button>
     </div>
     <div class="compass-command"><label class="dial-label" for="vesselHeadingInput">USV TRUE HDG COMMAND · LINKED TO COMPASS
       <input id="vesselHeadingInput" type="range" min="0" max="359" step="1" value="90" aria-label="USV command heading, linked to the compass command" /></label><span id="vesselCommandValue" class="cmd-value">090°</span></div>
     <div class="layer-picks" role="group" aria-label="3D engineering visualization layers">
       <label><input type="checkbox" data-layer="thrust" checked>Thrust vectors</label>
       <label><input type="checkbox" data-layer="axes" checked>Body axes</label>
       <label><input type="checkbox" data-layer="environment" checked>Wind / current</label>
       <label><input type="checkbox" data-layer="estimate" checked>Truth / nav estimate</label>
       <label><input type="checkbox" data-layer="route" checked>Waypoint bearing</label>
       <label><input type="checkbox" data-layer="wake" checked>Stylized wake</label>
       <label><input type="checkbox" data-layer="wireframe">Mesh edges</label>
     </div>
   </section>
   <section class="twin-frame compass-frame" id="compassFrame" hidden aria-label="Expandable Plotly true-north compass and calibration laboratory">
     <div class="twin-header"><strong>🧭 3D TRUE-HEADING COMPASS</strong><span>ENU · TRUE NORTH REFERENCE</span></div>
     <div class="camera-bar compass-tools" role="group" aria-label="Linked 3D camera controls">
       <button type="button" class="action selected" id="lock3D">🔗 LOCK 3D VIEWS</button>
       <button type="button" class="action" id="align3D">⊙ ALIGN VIEWS</button>
       <span class="tool-label" id="cameraSyncStatus">CAMERA LINKED</span>
     </div>
     <div id="compassPlot" aria-label="3D Plotly compass. Drag to orbit both views when linked; click a rim marker to command heading."></div>
     <canvas id="compassFallback" hidden aria-label="True heading compass fallback when Plotly WebGL is unavailable"></canvas>
     <div class="compass-status" id="compassStatus" role="status">Initializing compass...</div>
     <div class="compass-command">
       <label class="dial-label" for="compassHeadingInput">COMMAND TRUE HDG · DRAG TO STEER THROUGH AUTOPILOT
         <input id="compassHeadingInput" type="range" min="0" max="359" step="1" value="90" aria-label="Command a true heading; vessel responds through heading-hold dynamics" />
       </label>
       <span class="cmd-value" id="compassCommandValue">090°</span>
     </div>
     <div class="compass-note" id="compassReadout">CYAN: TRUE BOW · GREEN: MAGNETIC ESTIMATE · AMBER: COMMAND</div>
     <div class="magnetic-lab" aria-label="Compass calibration and linear algebra laboratory">
       <div class="magnetic-lab-title"><strong>MAGNETIC CALIBRATION & MATRIX DIAGNOSTICS</strong>
         <label><input type="checkbox" id="magCalibrationInput" checked /> Apply 3×3 calibration</label></div>
       <div class="magnetic-lab-sliders">
         <label>Hard-iron bias <span id="hardIronInputValue">0 µT</span><input id="hardIronInput" type="range" min="0" max="20" value="0" step="1"/></label>
         <label>Soft-iron distortion <span id="softIronInputValue">0%</span><input id="softIronInput" type="range" min="0" max="30" value="0" step="1"/></label>
         <label>Magnetic declination <span id="declinationInputValue">0°</span><input id="declinationInput" type="range" min="-20" max="20" value="0" step="1"/></label>
       </div>
       <div class="compass-note" id="magneticReadout">Initializing magnetometer…</div>
       <div class="compass-note" id="matrixReadout">15×15 covariance initializing…</div>
       <label class="allocation-priority">Control-allocation weighting
         <select id="allocationPriorityInput" aria-label="Thruster allocation priority">
           <option value="balanced">Balanced surge / yaw</option>
           <option value="heading">Prioritize heading moment</option>
           <option value="surge">Prioritize forward thrust</option>
         </select>
       </label>
       <div class="compass-note" id="allocatorReadout">Twin-thruster matrix initializing…</div>
     </div>
     <div class="compass-note">Orbit/zoom either Plotly scene to move both cameras when linked. Camera movement does not steer the vessel. Use the heading command to turn the USV physically.</div>
   </section>
   </div>
   </div>
   <div class="toolbar" role="group" aria-label="Autonomy and mission controls">
     <span class="tool-label">CONTROL</span>
     <button class="action selected" data-mode="manual">Manual</button>
     <button class="action" data-mode="heading">Heading hold</button>
     <button class="action" data-mode="auto">Waypoint auto</button>
     <span class="flex-space"></span>
     <button class="action" id="plan">＋ Add WP</button>
     <button class="action" id="demo">Demo route</button>
     <button class="action" id="clear">Clear</button>
   </div>
   <div class="helm">
     <div class="helm-left"><div id="controlMount"></div><div><div class="helm-title">Remote helm</div><div class="helm-mode" id="helmMode">Joystick</div><div class="helm-sub" id="helmHelp">Up/down: ahead/astern. Left/right: differential thrust. Manual control only.</div><button class="action" id="helmSwitch" style="margin-top:8px">Use D-pad</button></div></div>
     <div class="right-actions">
       <button class="action" id="lamp">💡 Lamp</button>
       <button class="action" id="sound" title="Toggle audio feedback">🔊 Sound</button>
       <button class="action" id="gnss">GNSS ON</button>
       <button class="action" id="c2">C2 ON</button>
       <button class="action warn" id="stop">STOP / IDLE</button>
       <button class="action good" id="resume">RESUME</button>
       <button class="action" id="pause">⏸ Pause</button>
       <button class="action" id="reset">↻ Reset</button>
     </div>
   </div>
   <section class="mission-screen" id="missionScreen">
     <div class="analytics-head"><strong>MISSION WORKSPACE · ROUTE MANAGEMENT</strong><span class="info-inline">Switch to Add WP, then tap the 2D chart</span></div>
     <div class="toolbar" aria-label="Route presets and editing">
       <button class="action" id="undoWp">↶ Undo last waypoint</button>
       <button class="action" id="safeRoute">⛨ Clearance route</button>
       <button class="action" id="importTrigger">↑ Import mission JSON</button>
       <input type="file" accept=".json,application/json" id="importFile" hidden>
       <button class="action" id="presetCalm">Calm water</button>
       <button class="action" id="presetWind">Crosswind</button>
       <button class="action" id="presetDenied">GNSS challenge</button>
     </div>
     <p class="info-inline">Waypoints use simulated East/North meters. Segment checks exclude moving traffic, navigation uncertainty and full vessel maneuverability. No bathymetric depth or certified collision avoidance. Mission import resets the browser scenario.</p>
   </section>
   <section class="challenge-screen" id="challengeScreen">
     <div class="analytics-head"><strong>MARITIME OPERATIONS CHALLENGE</strong><span class="badge">ILLUSTRATIVE TRAINING ONLY</span></div>
     <p class="info-inline">Complete marine objectives using the live USV model. Navigation gates are crossed without contact; inspections require the simulated searchlight, position and heading; station-keeping tests position hold; resilience tests supervised recovery.</p>
     <div class="challenge-config">
       <label>MISSION TYPE<select id="challengeKind"><option value="channel">Channel navigation</option><option value="inspection">Aid-to-navigation inspection</option><option value="station">Station-keeping</option><option value="resilience">Navigation resilience</option></select></label>
       <label>DIFFICULTY<select id="challengeLevel"><option value="basic">Basic</option><option value="standard" selected>Standard</option><option value="advanced">Advanced</option></select></label>
       <label>HELM AUTHORITY<select id="challengePilot"><option value="manual">Manual</option><option value="auto" selected>Autonomous</option></select></label>
       <button class="action good" id="challengeLaunch">▶ Launch challenge</button>
       <button class="action" id="challengeRestart">↻ Retry</button>
       <button class="action" id="challengeExport">↓ Results JSON</button>
     </div>
     <div class="challenge-details">
       <div><strong id="challengeBrief">Pass marked channel gates without contact.</strong><span id="challengeInstructions">Select a mission and launch to activate objective tracking.</span></div>
       <div class="challenge-live"><div id="challengeState">NOT STARTED</div><strong id="challengeScore">0 / 100</strong></div>
     </div>
     <div class="challenge-metrics">
       <div class="metric"><strong id="challengeProgress">0 / 0</strong><small>Objectives completed</small></div>
       <div class="metric"><strong id="challengeSafety">100</strong><small>Safety subscore</small></div>
       <div class="metric"><strong id="challengeAccuracy">—</strong><small>Tracking subscore</small></div>
       <div class="metric"><strong id="challengeClock">0.0 s</strong><small>Elapsed challenge time</small></div>
     </div>
     <div class="challenge-result" id="challengeResult" hidden></div>
     <div class="info-inline" style="margin-top:9px">Training score weights: objectives 35%, safety 25%, accuracy 15%, energy 15%, time 10%. Contacts or boundary violations fail the scenario regardless of points. Score does not establish real-world safety or COLREG compliance.</div>
   </section>
   <section class="analysis-screen" id="analysisScreen">
     <div class="analytics-head"><strong>LIVE ENGINEERING ANALYTICS</strong><label class="info-inline">Trace <select id="analyticTrace"><option value="speed">Actual / commanded speed</option><option value="error">Navigation error / predicted uncertainty</option><option value="power">Electrical power / motor temperature</option><option value="heading">Heading and yaw rate</option><option value="attitude">Roll / pitch (6-DOF)</option><option value="heave">Heave and wave elevation (6-DOF)</option></select></label></div>
     <div class="analysis-grid">
       <div class="metric"><strong id="aRmse">—</strong><small>Navigation RMSE · m</small></div>
       <div class="metric"><strong id="aXte">—</strong><small>Maximum |XTE| · m</small></div>
       <div class="metric"><strong id="aNis">—</strong><small>GNSS accepted / rejected</small></div>
       <div class="metric"><strong id="aEndurance">—</strong><small>Estimated endurance · min</small></div>
     </div>
     <canvas id="analysisCanvas" class="analysis-canvas" width="1100" height="300" aria-label="Time-series engineering traces"></canvas>
     <div class="toolbar"><button class="action" id="analysisCSV">↓ Export full telemetry</button><button class="action" id="analysisJSON">↓ Export mission JSON</button><span class="info-inline">Open the Streamlit Engineering Analysis page for uploaded replay, benchmark maneuvers, and Monte Carlo runs.</span></div>
   </section>
   <div class="toast" id="toast" aria-live="polite"></div>
 </div>
 <aside class="aside">
   <section class="panel">
     <h2 class="panel-title">Vessel telemetry</h2>
     <div class="metrics">
       <div class="metric"><strong id="sog">0.00</strong><small>Speed over ground · m/s</small></div>
       <div class="metric"><strong id="hdg">090°</strong><small>True heading</small></div>
       <div class="metric"><strong id="stw">0.00</strong><small>Speed through water · m/s</small></div>
       <div class="metric"><strong id="battery">100.0%</strong><small>Usable battery</small></div>
     </div>
     <div class="telemetry-row"><span>Surge / sway</span><b id="vel">0.00 / 0.00</b></div>
     <div class="telemetry-row"><span>Yaw rate</span><b id="yawrate">0.0 °/s</b></div>
     <div class="telemetry-row"><span>Heave z / vertical speed</span><b id="heave">0.00 m / 0.00 m/s</b></div>
     <div class="telemetry-row"><span>Roll / pitch</span><b id="attitude">0.0° / 0.0°</b></div>
     <div class="telemetry-row"><span>Roll / pitch rate</span><b id="attitudeRate">0.0 / 0.0 °/s</b></div>
     <div class="telemetry-row"><span>Regular wave surface</span><b id="waveReadout">0.00 m</b></div>
     <div class="telemetry-row"><span>Hydrostatic proxy (draft / GMt)</span><b id="hydroReadout">—</b></div>
     <div class="telemetry-row"><span>Port / starboard thrust</span><b id="motors">0 / 0 N</b></div>
     <div class="telemetry-row"><span>Power consumption</span><b id="power">25 W</b></div>
     <div class="telemetry-row"><span>Useful thrust / battery losses</span><b id="energyAudit">0 / 0 W</b></div>
     <div class="telemetry-row"><span>Thrust derating / numerical guards</span><b id="physicsGuard">100% / 0</b></div>
     <div class="telemetry-row"><span>Course over ground</span><b id="cog">—</b></div>
     <div class="telemetry-row"><span>Port / starboard RPM proxy</span><b id="rpm">0 / 0</b></div>
     <div class="telemetry-row"><span>DC bus / motor temperature</span><b id="electrical">25.2 V / 20°C</b></div>
   </section>
   <section class="panel">
     <h2 class="panel-title">Navigation & autonomy</h2>
     <div class="telemetry-row"><span>Active control</span><b id="activeMode">MANUAL</b></div>
     <div class="telemetry-row"><span>Track error</span><b id="xte">—</b></div>
     <div class="telemetry-row"><span>Waypoint distance</span><b id="wpDist">—</b></div>
     <div class="telemetry-row"><span>Nav position uncertainty</span><b id="uncert">1.2 m (1σ)</b></div>
     <div class="telemetry-row"><span>GNSS fix age</span><b id="gnssAge">0.0 s</b></div>
     <div class="telemetry-row"><span>Nearest hazard / CPA</span><b id="risk">—</b></div>
     <div class="telemetry-row"><span>Filter / GNSS innovation</span><b id="filterStatus">—</b></div>
     <div class="telemetry-row"><span>True / estimated deviation</span><b id="navError">—</b></div>
     <div class="health"><span class="badge" id="gnssHealth">GNSS NOMINAL</span><span class="badge" id="c2Health">C2 LINK OK</span></div>
     <div class="heading-adjust"><span class="tool-label">HDG SET</span><button class="action" id="hdgMinus">−15°</button><strong id="targetHdg">090°</strong><button class="action" id="hdgPlus">+15°</button></div>
     <div class="note" style="margin-top:8px">Position estimate can use either a four-state EKF with innovation gating or legacy complementary updates.</div>
   </section>
   <section class="panel">
     <h2 class="panel-title">Mission analytics</h2>
     <canvas id="trend" width="500" height="170" aria-label="Speed over ground and commanded speed history"></canvas>
     <div class="telemetry-row"><span>Mission progress</span><b id="progress">0 / 4</b></div>
     <div class="telemetry-row"><span>Energy used</span><b id="energy">0.0 Wh</b></div>
     <div class="telemetry-row"><span>Distance traveled</span><b id="distance">0.0 m</b></div>
     <div class="telemetry-row"><span>Collision events</span><b id="collisions">0</b></div>
     <div class="toolbar" style="margin-top:9px"><button class="action" id="csv">↓ CSV telemetry</button><button class="action" id="json">↓ JSON scenario</button></div>
     <h2 class="panel-title" style="margin:12px 0 5px">Event log</h2><div class="logs" id="logs">System initialized</div>
   </section>
 </aside>
</div>
<div class="footer"><span>Research visualization. Optional six-axis motions use uncalibrated geometric hydrostatics and illustrative wave excitation; 3-DOF modes remain planar. Stylized wake is not CFD. No certified autonomy, validated marine coefficients, or COLREG-compliant avoidance.</span><span>3/6-DOF model structures informed by <a href="https://www.fossen.biz/html/marineCraftModel.html" target="_blank" rel="noopener noreferrer">Fossen marine-craft dynamics</a>.</span></div>
</div>
<script id="cfg" type="application/json">__CFG_JSON__</script>
<script>__SIXDOF_ENGINE__</script>
<script>__POWERTRAIN_ENGINE__</script>
<script>__LINEAR_ALGEBRA_ENGINE__</script>
<script>__BLUEVECTOR_BROWSER_JS__</script>
</body>
</html>'''


# -----------------------------------------------------------------------------
# DEPENDENCY INJECTION / COMPOSITION ROOT
# The live browser document can be created without importing or invoking Streamlit.
# A future React, FastAPI or Electron host can use this contract and JS sources.
# -----------------------------------------------------------------------------
def build_browser_document(settings: Mapping[str, Any]) -> str:
    """Compile self-contained HTML from the portable simulation resources."""
    config = build_browser_config(settings)
    config_text = json.dumps(config, separators=(",", ":")).replace("<", "\\u003c")
    # The engines and controller are independent resource slots, not intermixed
    # with Streamlit widget code. Their order is deliberate.
    return (BROWSER_PAGE_TEMPLATE
            .replace("__BLUEVECTOR_CSS__", BROWSER_CSS)
            .replace("__CFG_JSON__", config_text)
            .replace("__SIXDOF_ENGINE__", SIXDOF_ENGINE)
            .replace("__POWERTRAIN_ENGINE__", POWERTRAIN_ENGINE)
            .replace("__LINEAR_ALGEBRA_ENGINE__", LINEAR_ALGEBRA_ENGINE)
            .replace("__BLUEVECTOR_BROWSER_JS__", BROWSER_CONTROLLER_JS))


def main() -> None:
    """Streamlit host only. No physics logic belongs in this function."""
    st.set_page_config(
        page_title="BLUEVECTOR | Autonomous USV Digital Twin",
        page_icon="🚤",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """<style>
        .block-container {padding-top: 0.75rem; padding-bottom: 1rem; max-width: 1700px;}
        [data-testid="stSidebar"] {background: #091b30;}
        </style>""",
        unsafe_allow_html=True,
    )
    workspace, controls = render_streamlit_controls()
    if workspace == "Live Simulation":
        components.html(build_browser_document(controls), height=1640, scrolling=True)
        st.caption(
            "BLUEVECTOR V10.1 | Combined USV and 3D compass · six-axis buoyancy, "
            "midpoint integration, current-relative terms, Plotly and Three.js. "
            "Hydrodynamic properties remain illustrative and uncalibrated."
        )
    else:
        render_engineering_analysis()


if __name__ == "__main__":
    main()
