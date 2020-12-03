/************************************************************************************************/
/*** Topic: Wang-Buzsaki model with Runge-Kutta 4th Order Method for one neuron    Ali-Seif   ***/
/*** Version Release 17.12 rev 11256                                                          ***/
/*** Date: 11/10/2020                                                                         ***/
/*** Code implemented in CodeBlocks C++ compiler (v. 17.12),                                  ***/
/*** MSI: PX60 6QD/ DDR4                                                                      ***/
/*** Run under a Intel® Core™ i7-6700HQ CPU @ 2.60GHz × 64 based processor with 16 GB RAM     ***/
/************************************************************************************************/
#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;                                            //for Standard program

//##############################################################
//####                                                      ####
//####                 Create class Neuron                  ####
//####                                                      ####
//##############################################################
class Neuron
{
	private:
		double Gna = 120.0;
		double Gk = 36.0;
		double Gl = 0.3;
		double El = -49.42;
		double Ena = 55.17;
		double Ek = -72.14;
        double Cm = 1.0;
        double Iapp= 6.5;
	public:
        void onedt(void);
		double alpha_n(double);
		double beta_n(double);
	  	double alpha_m(double);
	  	double beta_m(double);
	  	double alpha_h(double);
	  	double beta_h(double);
	  	double n_inf(double);
	  	double m_inf(double);
	  	double h_inf(double);
	  	double INa1(double,double,double);
	  	double IK1(double,double);
	  	double Il1(double);
	  	double dvdt(double,double,double,double,double);
	  	double dndt(double,double,double);
        double dhdt(double,double,double);
        double dmdt(double,double,double);
	  	double rk4thOrder_v(double,double,double,double,double,double);
	  	double rk4thOrder_n(double,double,double,double);
	  	double rk4thOrder_h(double,double,double,double);
        double rk4thOrder_m(double,double,double,double);
        double t_final=300;
        double dt = 0.01;
	  	double t0;
  		double v,n,m,h;
};
//_________________________________Calculate alpha and betas_________________________________//

double Neuron::alpha_n(double v){return   0.01*(v+50)/(1-exp(-(v+50)/10));}
double Neuron::beta_n(double v){return   0.125*exp(-(v+60)/80);}
double Neuron::alpha_m(double v){return   0.1*(v+35)/(1-exp(-(v+35)/10));}
double Neuron::beta_m(double v){return   4.0*exp(-0.0556*(v+60));}
double Neuron::alpha_h(double v){return   0.07*exp(-0.05*(v+60));}
double Neuron::beta_h(double v){return   1/(1+exp(-(0.1)*(v+30)));}

//__________________________Calculate infinite activation variables__________________________//

double Neuron::n_inf(double v){return alpha_n(v)/(alpha_n(v)+beta_n(v));}
double Neuron::h_inf(double v){return alpha_h(v)/(alpha_h(v)+beta_h(v));}
double Neuron::m_inf(double v){return alpha_m(v)/(alpha_m(v)+beta_m(v));}

//__________________________________Calculation of currents__________________________________//

double Neuron::INa1(double v,double h,double m) {return Gna*h*pow(m,3)*(v-Ena);}
double Neuron::IK1(double v,double n) {return Gk*pow(n,4)*(v-Ek);}
double Neuron::Il1(double v) {return Gl*(v-El);}

//___________________________________Differential Equations__________________________________//

double Neuron::dvdt(double t, double v,double n,double h,double m){return  (1/Cm)*(Iapp -(INa1(v,h,m)+IK1(v,n)+Il1(v)));}
double Neuron::dndt(double t,double n, double v){return  ((alpha_n(v)*(1-n))-beta_n(v)*n);}
double Neuron::dhdt(double t, double h, double v){return   ((alpha_h(v)*(1-h))-beta_h(v)*h);}
double Neuron::dmdt(double t, double m, double v){return   ((alpha_m(v)*(1-m))-beta_m(v)*m);}

//__________________________________Runge-Kutta calculations_________________________________//

double Neuron::rk4thOrder_v(double t0, double v, double dt,double n,double h,double m) {
    double  k1, k2, k3, k4;
            k1=     dt*dvdt(t0, v,n,h,m);
            k2=     dt*dvdt((t0+dt/2), (v+k1/2),n,h,m);
            k3=     dt*dvdt((t0+dt/2), (v+k2/2),n,h,m);
            k4=     dt*dvdt((t0+dt), (v+k3),n,h,m);
            v=      v+double((1.0/6.0)*(k1+2*k2+2*k3+k4));
   return   v;}
double Neuron::rk4thOrder_n(double t0, double v, double dt, double n) {
    double  k1, k2, k3, k4;
            k1=     dt*dndt(t0, n,v);
            k2=     dt*dndt((t0+dt/2), (n+k1/2),v);
            k3=     dt*dndt((t0+dt/2), (n+k2/2),v);
            k4=     dt*dndt((t0+dt), (n+k3),v);
            n=      n+double((1.0/6.0)*(k1+2*k2+2*k3+k4));
   return   n;}

double Neuron::rk4thOrder_h(double t0, double v, double dt,double h) {
    double  k1, k2, k3, k4;
            k1=     dt*dhdt(t0, h,v);
            k2=     dt*dhdt((t0+dt/2), (h+k1/2),v);
            k3=     dt*dhdt((t0+dt/2), (h+k2/2),v);
            k4=     dt*dhdt((t0+dt), (h+k3),v);
            h=      h+double((1.0/6.0)*(k1+2*k2+2*k3+k4));
    return   h;}
double Neuron::rk4thOrder_m(double t0, double v, double dt,double m) {
    double  k1, k2, k3, k4;
            k1=     dt*dmdt(t0, m,v);
            k2=     dt*dmdt((t0+dt/2), (m+k1/2),v);
            k3=     dt*dmdt((t0+dt/2), (m+k2/2),v);
            k4=     dt*dmdt((t0+dt), (m+k3),v);
            m=      m+double((1.0/6.0)*(k1+2*k2+2*k3+k4));
    return   m;}

//__________________________________Runge-Kutta calculations_________________________________//
void Neuron::onedt(){
    v=rk4thOrder_v(t0, v, dt,n,h,m);
    n=rk4thOrder_n(t0,v, dt ,n);
    h=rk4thOrder_h(t0, v, dt ,h);
    m=rk4thOrder_m(t0, v, dt ,m);
    }









//_______________________________________________________________________________________\\
//_____________                                                             _____________\\
//_____________                                      @                      _____________\\
//_____________           @@       @@       @            @@     @           _____________\\
//_____________           @ @     @ @      @ @       @   @ @    @           _____________\\
//_____________           @  @   @  @     @   @      @   @  @   @           _____________\\
//_____________           @   @@@   @    @@@@@@@     @   @   @  @           _____________\\
//_____________           @    @    @   @       @    @   @    @ @           _____________\\
//_____________           @         @  @         @   @   @     @@           _____________\\
//_______________________________________________________________________________________

int main() {


    Neuron neuron;
    ofstream temp("temp.txt", ios::out | ios::trunc);

    neuron.v=-20.0;
    neuron.n=neuron.n_inf(neuron.v);
    neuron.h=neuron.h_inf(neuron.v);
    neuron.m=neuron.m_inf(neuron.v);

    for ( neuron.t0=neuron.dt ; neuron.t0<=neuron.t_final ; neuron.t0=neuron.t0 + neuron.dt){

        neuron.onedt();
        temp << neuron.t0 << '\t' <<neuron.v<< endl;

    }




    temp.close();
    cout << "\nFinish" << endl;

    return 0;
}
