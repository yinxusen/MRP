public class MRPPara {

	public double lambdaF;
	public double lambdaA;
	public double lambdaN;
	public double lambdaM;
	public double lambdaRU;
	public double lambdaRV;
	public double lambdaRZ;
	public double lambdaRW;
	public double lambdaRM;
	public double lambdaRr;
	public double lambdaRs;
	public double lambdaRH;
	public static final double good = 0.01;
	public static final int T = 1;

	public double lr;
	public double zlr;

	public MRPPara(double lf, double la, double ln, double lm, double lru,
			double lrv, double lrz, double lrw, double lrm, double lrr,
			double lrs, double lr, double lrh) {
		this.lambdaF = lf;
		this.lambdaA = la;
		this.lambdaN = ln;
		this.lambdaM = lm;
		this.lambdaRU = lru;
		this.lambdaRV = lrv;
		this.lambdaRZ = lrz;
		this.lambdaRW = lrw;
		this.lambdaRM = lrm;
		this.lambdaRr = lrr;
		this.lambdaRs = lrs;
		this.lambdaRH = lrh;
		this.lr = lr;
	}

	public MRPPara() {
		this.lambdaF = 0.3;
		this.lambdaA = 0.3;
		this.lambdaN = 0.2;
		this.lambdaM = 0.2;
		this.lambdaRU = 0.002;
		this.lambdaRV = 0.002;
		this.lambdaRZ = 0.01;
		this.lambdaRW = 0.002;
		this.lambdaRM = 0.002;
		this.lambdaRr = 0.1;
		this.lambdaRs = 0.1;
		this.lambdaRH = 0.01;
		this.lr = 0.05;
		this.zlr = 5;
		
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
