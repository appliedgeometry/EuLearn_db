# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007


lissajous_c="""__host__ __device__ double curve_x(double t){//lissajous
		double p;
		p=cos(@knx*t + @phi_x);
		
		return(p);}

		__host__ __device__ double curve_y(double t){//lissajous
		double p;
		
		p=cos(@kny*t + @phi_y);
		
		return(p);}


		__host__ __device__ double curve_z(double t){//lissajous
		double p;
		
		p=cos(@knz*t + @phi_z);
		return(p);}
"""



fibonacci_c="""__host__ __device__ double curve_x(double t){//fibbonacci
		double p;
		//int A=0.5;
		p=cos(@knx*t + @phi_x);
		
		return(p);}

		__host__ __device__ double curve_y(double t){//fibbonacci
		double p;
		//int A=0.5;
		
		p=cos(@kny*t + @phi_y);
		
		return(p);}

		__host__ __device__ double curve_z(double t){//fibbonacci
		double p;
		int A=0.5;
		
		p=A*cos(@knz*t + @phi_z)+A*sin(@kny*t + @phi_y);
		return(p);}




"""


trefoil_c="""__host__ __device__ double curve_x(double t){//trefoil
		double p;
		p=sin(t)+2*sin(t*2);

		return(p);}

		__host__ __device__ double curve_y(double t){//trefoil
		double p;
		
		p=cos(t)-2*cos(t*2);
		
		return(p);}


		__host__ __device__ double curve_z(double t){//trefoil
		double p;
		
		p=-sin(3*t);
		return(p);}

"""


eight_c="""__host__ __device__ double curve_x(double t){//eight
		double p;
		p=(2 + cos(2*t))*cos(3*t);

		return(p);}

		__host__ __device__ double curve_y(double t){//eight
		double p;
		
		p=(2 + cos(2*t))*sin(3*t);
		
		return(p);}


		__host__ __device__ double curve_z(double t){//eight
		double p;
		
		p=sin(4*t);
		return(p);}

"""


circle_c="""__host__ __device__ double curve_x(double t){//circle
		double p;
		p=sin(t);
		
		return(p);}

		__host__ __device__ double curve_y(double t){//circle
		double p;
		
		p=cos(t);
		
		return(p);}

		__host__ __device__ double curve_z(double t){//circle
		double p;
		
		p=0;
		return(p);}


"""

