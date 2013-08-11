import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import loss.leastSquare;
import cern.jet.math.*;
import cern.colt.matrix.*;
import cern.colt.matrix.linalg.*;

public class MRP {

	// sparse index
	public class sIdx {
		int sId;
		double weight;

		public sIdx(int sId, double weight) {
			super();
			this.sId = sId;
			this.weight = weight;
		}
	}

	public static final double smooth = 0.01;

	int dimGF;
	int dimGA;
	int dimLatent;
	int dimFeatureF;
	int dimFeatureA;
	int dimBiFeatureF;
	int dimBiFeatureA;

	MRPPara para = null;

	double[][] H = null;
	double[][] U = null;
	double[][] V = null;
	double[][] Z = null;
	double[][] W = null;
	double[][] M = null;
	double[] R = null;
	double[] S = null;

	double[][] HH = null;
	double[][] YH = null;
	double[][] FM = null;

	leastSquare l2 = leastSquare.getInstance();

	List<sIdx>[] GF = null;
	List<sIdx>[] GA = null;
	List<sIdx>[] reverseGA = null;
	List<sIdx>[] featureF = null;
	List<sIdx>[] featureA = null;

	Map<String, Integer> userMap = null;
	Map<String, Integer> attrMap = null;
	Map<String, Integer> userFMap = null;
	Map<String, Integer> attrFMap = null;

	private static int userCount = 0;
	private static int attrCount = 0;
	private static int userFCount = 0;
	private static int attrFCount = 0;

	UserIndex useridx = null;
	AttrIndex attridx = null;
	UserFIndex userfidx = null;
	AttrFIndex attrfidx = null;

	public void computeFM() {
		if (this.FM == null) {
			this.FM = new double[this.dimGF][this.dimFeatureA];
		}
		for (int i = 0; i < dimGF; i++) {
			for (int j = 0; j < dimFeatureA; j++) {
				this.FM[i][j] = 0;
				double[] ff = sparse2dense(featureF[i], dimFeatureF);
				for (int k = 0; k < dimFeatureF; k++) {
					this.FM[i][j] += ff[k] * M[k][j];
				}
			}
		}
	}

	public void computeHH() {
		if (this.HH == null) {
			this.HH = new double[dimLatent][dimLatent];
		}
		for (int i = 0; i < dimLatent; i++) {
			for (int j = 0; j < dimLatent; j++) {
				this.HH[i][j] = 0;
				for (int k = 0; k < dimGF; k++) {
					this.HH[i][j] += H[k][i] * H[k][j];
				}
			}
		}
	}

	public double[] computeVectorNewY(int idx) {
		double[] y = sparse2dense(this.reverseGA[idx], this.dimGF);
		return plusVector(
				y,
				negVector(vectorMultiMatrix(
						sparse2dense(this.featureA[idx], dimFeatureA),
						trans(this.FM))));

	}

	public void computeYH() {
		if (this.YH == null) {
			this.YH = new double[dimGA][dimLatent];
		}
		double[][] transH = trans(H);
		for (int i = 0; i < dimGA; i++) {
			for (int j = 0; j < dimLatent; j++) {
				this.YH[i][j] = dot(computeVectorNewY(i), transH[j]);
			}
		}
	}

	public void computeReverseGA() {
		if (this.reverseGA == null) {
			this.reverseGA = new List[dimGA];
			for (int i = 0; i < dimGA; i++) {
				this.reverseGA[i] = new ArrayList<sIdx>();
			}
		}
		for (int i = 0; i < dimGF; i++) {
			for (sIdx s : this.GA[i]) {
				this.reverseGA[s.sId].add(new sIdx(i, s.weight));
			}
		}
	}

	public void init(int dgf, int dga, int dl, int dff, int dfa, int dbff,
			int dbfa) {
		this.para = new MRPPara();
		this.dimBiFeatureA = dbfa;
		this.dimBiFeatureF = dbff;
		this.dimFeatureA = dfa;
		this.dimFeatureF = dff;
		this.dimGA = dga;
		this.dimGF = dgf;
		this.dimLatent = dl;
		this.H = new double[dimGF][dimLatent];
		this.U = new double[dimGF][dimLatent];
		this.V = new double[dimGF][dimLatent];
		this.Z = new double[dimGA][dimLatent];
		this.W = new double[dimFeatureF][dimFeatureF];
		this.M = new double[dimFeatureF][dimFeatureA];
		this.R = new double[dimBiFeatureF];
		this.S = new double[dimBiFeatureA];
		this.GF = new ArrayList[dimGF];
		this.GA = new ArrayList[dimGF];
		this.featureF = new ArrayList[dimGF];
		this.featureA = new ArrayList[dimGA];
		for (int i = 0; i < dimGF; i++) {
			this.GF[i] = new ArrayList<sIdx>();
			this.featureF[i] = new ArrayList<sIdx>();
			this.GA[i] = new ArrayList<sIdx>();
		}
		for (int i = 0; i < dimGA; i++) {
			this.featureA[i] = new ArrayList<sIdx>();
		}
		this.userMap = new HashMap<String, Integer>();
		this.attrMap = new HashMap<String, Integer>();
		this.userFMap = new HashMap<String, Integer>();
		this.attrFMap = new HashMap<String, Integer>();

		useridx = new UserIndex();
		attridx = new AttrIndex();
		userfidx = new UserFIndex();
		attrfidx = new AttrFIndex();
	}

	/*
	 * We can init them using random function But local classifier is a better
	 * choice.
	 */
	public void initParameterAndLatentFactor() {
		for (int i = 0; i < dimGF; i++) {
			for (int j = 0; j < dimLatent; j++) {
				H[i][j] = Math.random() * 1;
				U[i][j] = Math.random() * 1;
				V[i][j] = Math.random() * 1;
			}
		}
		for (int i = 0; i < dimGA; i++) {
			for (int j = 0; j < dimLatent; j++) {
				Z[i][j] = Math.random() * 1;
			}
		}
		for (int i = 0; i < dimFeatureF; i++) {
			for (int j = 0; j < dimFeatureF; j++) {
				W[i][j] = Math.random() * 1;
			}
		}
		for (int i = 0; i < dimFeatureF; i++) {
			for (int j = 0; j < dimFeatureA; j++) {
				M[i][j] = Math.random() * 1;
			}
		}
	}

	public void initParameterAndLatentFactor(String fName,
			double[][] container, double dim1, double dim2)
			throws NumberFormatException, IOException {
		FileInputStream file = new FileInputStream(new File(fName));
		System.out.println("read " + fName);
		BufferedReader reader = new BufferedReader(new InputStreamReader(file,
				"UTF-8"));
		int cnt = 0;
		String tempString = null;
		while ((tempString = reader.readLine()) != null) {
			cnt++;
			if (cnt >= dim1)
				break;
			String[] strArray = tempString.split(",");
			for (int j = 0; j < dim2; j++) {
				container[cnt][j] = Double.parseDouble(strArray[j]);
			}
		}
		reader.close();
	}

	public void initParameterAndLatentFactor(String fH, String fM, String fU,
			String fV, String fW, String fZ) throws NumberFormatException,
			IOException {
		this.initParameterAndLatentFactor(fH, H, dimGF, dimLatent);
		this.initParameterAndLatentFactor(fM, M, dimFeatureF, dimFeatureA);
		this.initParameterAndLatentFactor(fU, U, dimGF, dimLatent);
		this.initParameterAndLatentFactor(fV, V, dimGF, dimLatent);
		this.initParameterAndLatentFactor(fW, W, dimFeatureF, dimFeatureF);
		this.initParameterAndLatentFactor(fZ, Z, dimGA, dimLatent);
	}

	public interface Index {
		public int Index(String str);
	}

	public class UserIndex implements Index {
		public int Index(String str) {
			if (userMap.containsKey(str)) {
				return userMap.get(str);
			} else {
				userMap.put(str, MRP.userCount);
				return MRP.userCount++;
			}
		}
	}

	public class UserFIndex implements Index {
		public int Index(String str) {
			if (userFMap.containsKey(str)) {
				return userFMap.get(str);
			} else {
				userFMap.put(str, MRP.userFCount);
				return MRP.userFCount++;
			}
		}
	}

	public class AttrIndex implements Index {
		public int Index(String str) {
			if (attrMap.containsKey(str)) {
				return attrMap.get(str);
			} else {
				attrMap.put(str, MRP.attrCount);
				return MRP.attrCount++;
			}
		}
	}

	public class AttrFIndex implements Index {
		public int Index(String str) {
			if (attrFMap.containsKey(str)) {
				return attrFMap.get(str);
			} else {
				attrFMap.put(str, MRP.attrFCount);
				return MRP.attrFCount++;
			}
		}
	}

	private void readData(String f, List<sIdx>[] container, Index idxP,
			Index idxM) throws NumberFormatException, IOException {
		BufferedReader reader = null;
		FileInputStream file = new FileInputStream(new File(f));
		System.out.println("read " + f);
		reader = new BufferedReader(new InputStreamReader(file, "UTF-8"));
		String tempString = null;

		int cnt = 0;
		while ((tempString = reader.readLine()) != null) {
			cnt++;
			if (cnt % 1000 == 0) {
				System.out.println("read");
			}
			String[] strArray = tempString.split(",");
			Integer primaryId = idxP.Index(strArray[0]);
			Integer minorId = idxM.Index(strArray[1]);
			// System.out.println(primaryId);
			double weight = Double.parseDouble(strArray[2]);
			if (weight > 10000) {
				System.out.println("shit!");
			}
			container[primaryId].add(new sIdx(minorId, weight));
		}
		reader.close();
	}

	public double dot(double[] a, double[] b) {
		assert (a.length == b.length);
		double retval = 0.0;
		for (int i = 0; i < a.length; i++) {
			retval += a[i] * b[i];
		}
		return retval;
	}

	public double[][] outerDot(double[] a, double[] b) {
		int m = a.length;
		int n = b.length;
		double[][] retval = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				retval[i][j] = a[i] * b[j];
			}
		}
		return retval;
	}

	public double[][] dotMulti(double a, double[][] b) {
		int m = b.length;
		int n = b[0].length;
		double[][] retval = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				retval[i][j] = b[i][j] * a;
			}
		}
		return retval;
	}

	public double[] dotVector(double a, double[] b) {
		int m = b.length;
		double[] retval = new double[m];
		for (int i = 0; i < m; i++) {
			retval[i] = b[i] * a;
		}
		return retval;
	}

	public double[][] trans(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] b = new double[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				b[j][i] = a[i][j];
			}
		}
		return b;
	}

	public double[] vectorMultiMatrix(double[] a, double[][] b) {
		assert (a.length == b.length);
		double[] retval = new double[b[0].length];
		double[][] bT = trans(b);
		for (int i = 0; i < b[0].length; i++) {
			retval[i] = dot(a, bT[i]);
		}
		return retval;
	}

	public double[] sparse2dense(List<sIdx> ls, int dim) {
		double retval[] = new double[dim]; // initial zeros?
		for (sIdx s : ls) {
			retval[s.sId] = s.weight;
		}
		return retval;
	}

	public double[][] plusMat(double[][] a, double[][] b) {
		assert (a.length == b.length && a[0].length == b[0].length);
		int m = a.length;
		int n = a[0].length;
		double[][] retval = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				retval[i][j] = a[i][j] + b[i][j];
			}
		}
		return retval;
	}

	public double[] plusVector(double[] a, double[] b) {
		assert (a.length == b.length);
		int m = a.length;
		double[] retval = new double[m];
		for (int i = 0; i < m; i++) {
			retval[i] = a[i] + b[i];
		}
		return retval;
	}

	public double[][] negMat(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] retval = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				retval[i][j] = -a[i][j];
			}
		}
		return retval;
	}

	public double[] negVector(double[] a) {
		int m = a.length;
		double[] retval = new double[m];
		for (int i = 0; i < m; i++) {
			retval[i] = -a[i];
		}
		return retval;
	}

	/* Don't consider pairwise features now */
	public double predictFriendship(int i, int ii) {
		double[] x_i = sparse2dense(this.featureF[i], this.dimFeatureF);
		double[] x_ii = sparse2dense(this.featureF[ii], this.dimFeatureF);
		double retval = dot(H[i], V[ii])
				+ dot(x_ii, vectorMultiMatrix(x_i, this.W));
		// double retval = dot(x_ii, vectorMultiMatrix(x_i, this.W));
//		if (retval > 1000) {
//			System.out.println("shit");
//		}
		return retval;
	}

	public double predictAttribute(int i, int ii) {
		double[] x_i = sparse2dense(this.featureF[i], this.dimFeatureF);
		double[] x_ii = sparse2dense(this.featureA[ii], this.dimFeatureA);
		return dot(H[i], Z[ii]) + dot(x_ii, vectorMultiMatrix(x_i, this.M));
	}

	public double predictFeatureF(int i, int ii) {
		return dot(H[i], U[ii]);
	}

	public interface updateMethod {
		public void update(List<sIdx> ls, int curIdx);

		public void updateH(List<sIdx> ls, int curIdx);

		public void updateU(List<sIdx> ls, int curIdx);

		public void updateV(List<sIdx> ls, int curIdx);

		public void updateZ(List<sIdx> ls, int curIdx);

		public void updateW(List<sIdx> ls, int curIdx);

		public void updateM(List<sIdx> ls, int curIdx);
	}

	public double fNorm(double[][] a) {
		double retval = 0.0;
		int m = a.length;
		int n = a[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				retval += a[i][j] * a[i][j];
			}
		}
		return Math.sqrt(retval) > MRP.smooth ? Math.sqrt(retval) : MRP.smooth;
	}

	public double l2Norm(double[] a) {
		double retval = 0.0;
		int m = a.length;
		for (int i = 0; i < m; i++) {
			retval += a[i] * a[i];
		}
		return Math.sqrt(retval) > MRP.smooth ? Math.sqrt(retval) : MRP.smooth;
	}

	public double sgn(double a) {
		return a >= 0 ? 1 : -1;
	}

	public double posPart(double a) {
		// System.err.println("work");
		return a >= 0 ? a : 0;
	}

	updateGF ugf = new updateGF();

	/* update H V W */
	class updateGF implements updateMethod {
		public void update(List<sIdx> ls, int curIdx) {

			/* update W */
			for (sIdx s : ls) {
				double[] x_i = sparse2dense(featureF[curIdx], dimFeatureF);
				double[] x_ii = sparse2dense(featureF[s.sId], dimFeatureF);
				double[][] dW1 = dotMulti(
						para.lambdaF
								* l2.getPartialDerivation(s.weight,
										predictFriendship(curIdx, s.sId)),
						outerDot(x_i, x_ii));
				assert (fNorm(W) != 0);
				double[][] dW2 = dotMulti(para.lambdaRW * 0.5
						* (1.0 / fNorm(W)), W);
				double[][] dW = plusMat(dW1, dW2);
				// gradient descent once
				W = plusMat(W, negMat(dotMulti(para.lr, dW)));
			}
			/* update H */
			for (sIdx s : ls) {
				double[] dH1 = dotVector(
						para.lambdaF
								* l2.getPartialDerivation(s.weight,
										predictFriendship(curIdx, s.sId)),
						V[s.sId]);
				assert (l2Norm(H[curIdx]) != 0);
				double[] dH2 = dotVector(para.lambdaRH
						* (1.0 / l2Norm(H[curIdx])), H[curIdx]);
				double[] dH = plusVector(dH1, dH2);
				// gradient descent once
				H[curIdx] = plusVector(H[curIdx],
						negVector(dotVector(para.lr, dH)));
			}
			/* update V */
			for (sIdx s : ls) {
				double[] dV1 = dotVector(
						para.lambdaF
								* l2.getPartialDerivation(s.weight,
										predictFriendship(curIdx, s.sId)),
						H[s.sId]);
				assert (l2Norm(V[curIdx]) != 0);
				double[] dV2 = dotVector(para.lambdaRV
						* (1.0 / l2Norm(V[curIdx])), V[curIdx]);
				double[] dV = plusVector(dV1, dV2);
				// gradient descent once
				V[curIdx] = plusVector(V[curIdx],
						negVector(dotVector(para.lr, dV)));
			}
		}

		@Override
		public void updateH(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update H */
			for (sIdx s : ls) {
				double[] dH1 = dotVector(
						para.lambdaF
								* l2.getPartialDerivation(s.weight,
										predictFriendship(curIdx, s.sId)),
						V[s.sId]);
				assert (l2Norm(H[curIdx]) != 0);
				double[] dH2 = dotVector(para.lambdaRH
						* (1.0 / l2Norm(H[curIdx])), H[curIdx]);
				double[] dH = plusVector(dH1, dH2);
				// gradient descent once
				H[curIdx] = plusVector(H[curIdx],
						negVector(dotVector(para.lr, dH)));
			}
		}

		@Override
		public void updateU(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateV(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			for (sIdx s : ls) {
				double[] dV1 = dotVector(
						para.lambdaF
								* l2.getPartialDerivation(s.weight,
										predictFriendship(curIdx, s.sId)),
						H[s.sId]);
				assert (l2Norm(V[curIdx]) != 0);
				double[] dV2 = dotVector(para.lambdaRV
						* (1.0 / l2Norm(V[curIdx])), V[curIdx]);
				double[] dV = plusVector(dV1, dV2);
				// gradient descent once
				V[curIdx] = plusVector(V[curIdx],
						negVector(dotVector(para.lr, dV)));
			}
		}

		@Override
		public void updateZ(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateW(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update W */
			for (sIdx s : ls) {
				double[] x_i = sparse2dense(featureF[curIdx], dimFeatureF);
				double[] x_ii = sparse2dense(featureF[s.sId], dimFeatureF);
				double[][] dW1 = dotMulti(
						para.lambdaF
								* l2.getPartialDerivation(s.weight,
										predictFriendship(curIdx, s.sId)),
						outerDot(x_i, x_ii));
				assert (fNorm(W) != 0);
				double[][] dW2 = dotMulti(para.lambdaRW * 0.5
						* (1.0 / fNorm(W)), W);
				double[][] dW = plusMat(dW1, dW2);
				// gradient descent once
				W = plusMat(W, negMat(dotMulti(para.lr, dW)));
			}
		}

		@Override
		public void updateM(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}
	}

	updateGA uga = new updateGA();

	/* update H Z M */
	class updateGA implements updateMethod {
		public void update(List<sIdx> ls, int curIdx) {
			/* update M */
			for (sIdx s : ls) {
				double[] x_i = sparse2dense(featureF[curIdx], dimFeatureF);
				double[] x_ii = sparse2dense(featureA[s.sId], dimFeatureA);
				double[][] dM1 = dotMulti(
						para.lambdaA
								* l2.getPartialDerivation(s.weight,
										predictAttribute(curIdx, s.sId)),
						outerDot(x_i, x_ii));
				assert (fNorm(M) != 0);
				double[][] dM2 = dotMulti(para.lambdaRM * 0.5
						* (1.0 / fNorm(M)), M);
				double[][] dM = plusMat(dM1, dM2);
				// gradient descent once
				M = plusMat(M, negMat(dotMulti(para.lr, dM)));
			}
			/* update H */
			for (sIdx s : ls) {
				double[] dH1 = dotVector(
						para.lambdaA
								* l2.getPartialDerivation(s.weight,
										predictAttribute(curIdx, s.sId)),
						Z[s.sId]);
				assert (l2Norm(H[curIdx]) != 0);
				double[] dH2 = dotVector(para.lambdaRH
						* (1.0 / l2Norm(H[curIdx])), H[curIdx]);
				double[] dH = plusVector(dH1, dH2);
				// gradient descent once
				H[curIdx] = plusVector(H[curIdx],
						negVector(dotVector(para.lr, dH)));
			}
			/* update Z */
			for (sIdx s : ls) {
				/* take one value, execute once for all dimension */
				for (int t = 0; t < 100; t++) {
					for (int i = 0; i < dimLatent; i++) {
						double[] F = sparse2dense(featureF[i], dimFeatureF);
						double[] A = sparse2dense(featureA[s.sId], dimFeatureA);
						double star = (s.weight
								- dot(A, vectorMultiMatrix(F, M))
								- dot(H[curIdx], Z[s.sId]) + H[curIdx][i]
								* Z[s.sId][i])
								* H[curIdx][i];
						double denominator = (H[curIdx][i] * H[curIdx][i]) > MRP.smooth ? (H[curIdx][i] * H[curIdx][i])
								: MRP.smooth;
						Z[s.sId][i] = sgn(star)
								* posPart(Math.abs(star) - para.lambdaRZ
										/ para.lambdaA) / denominator;
					}
				}
			}
		}

		@Override
		public void updateH(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update H */
			for (sIdx s : ls) {
				double[] dH1 = dotVector(
						para.lambdaA
								* l2.getPartialDerivation(s.weight,
										predictAttribute(curIdx, s.sId)),
						Z[s.sId]);
				assert (l2Norm(H[curIdx]) != 0);
				double[] dH2 = dotVector(para.lambdaRH
						* (1.0 / l2Norm(H[curIdx])), H[curIdx]);
				double[] dH = plusVector(dH1, dH2);
				// gradient descent once
				H[curIdx] = plusVector(H[curIdx],
						negVector(dotVector(para.lr, dH)));
			}
		}

		@Override
		public void updateU(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateV(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		public void updateZVectorwise(List<sIdx> ls, int curIdx) {
			/* update Z */
			double err = 0.0;
			double last_err = Double.MAX_VALUE;
			for (int t = 0; t < para.T; t++) {
				for (int i = 0; i < dimLatent; i++) {
					double star = YH[curIdx][i] - dot(HH[i], Z[curIdx])
							+ HH[i][i] * Z[curIdx][i];
					Z[curIdx][i] = sgn(star)
							* posPart(Math.abs(star) - 0.5 * para.lambdaRZ
									/ para.lambdaA) / HH[i][i];
					// if (HH[i][i] < sgn(star)
					// * posPart(Math.abs(star) - 0.5 * para.lambdaRZ
					// / para.lambdaA)) {
					// System.err.println(HH[i][i] + "\t" +Z[curIdx][i]);
					// }
				}
				err = errA();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + t);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
		}

		@Override
		public void updateZ(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update Z */
			double err = 0.0;
			double last_err = Double.MAX_VALUE;
			for (sIdx s : ls) {
				for (int i = 0; i < dimLatent; i++) {
					double[] F = sparse2dense(featureF[i], dimFeatureF);
					double[] A = sparse2dense(featureA[s.sId], dimFeatureA);
					double star = (s.weight - dot(A, vectorMultiMatrix(F, M))
							- dot(H[curIdx], Z[s.sId]) + H[curIdx][i]
							* Z[s.sId][i])
							* H[curIdx][i];
					double denominator = (H[curIdx][i] * H[curIdx][i]) > MRP.smooth ? (H[curIdx][i] * H[curIdx][i])
							: MRP.smooth;
					Z[s.sId][i] = sgn(star)
							* posPart(Math.abs(star) - 0.5 * para.lambdaRZ
									/ para.lambdaA) / denominator;
					// err = errA();
					// if ((err <= MRPPara.good) || (err > last_err)) {
					// // System.err.println("iter\t");
					// // break;
					// }
					// last_err = err;
					// System.out.println(err);
				}
			}
		}

		@Override
		public void updateW(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateM(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update M */
			for (sIdx s : ls) {
				double[] x_i = sparse2dense(featureF[curIdx], dimFeatureF);
				double[] x_ii = sparse2dense(featureA[s.sId], dimFeatureA);
				double[][] dM1 = dotMulti(
						para.lambdaA
								* l2.getPartialDerivation(s.weight,
										predictAttribute(curIdx, s.sId)),
						outerDot(x_i, x_ii));
				assert (fNorm(M) != 0);
				double[][] dM2 = dotMulti(para.lambdaRM * 0.5
						* (1.0 / fNorm(M)), M);
				double[][] dM = plusMat(dM1, dM2);
				// gradient descent once
				M = plusMat(M, negMat(dotMulti(para.lr, dM)));
			}
		}
	}

	updateFF uff = new updateFF();

	/* update H U */
	class updateFF implements updateMethod {
		public void update(List<sIdx> ls, int curIdx) {
			/* update H */
			for (sIdx s : ls) {
				double[] dH1 = dotVector(
						para.lambdaN
								* l2.getPartialDerivation(s.weight,
										predictFeatureF(curIdx, s.sId)),
						U[s.sId]);
				assert (l2Norm(H[curIdx]) != 0);
				double[] dH2 = dotVector(para.lambdaRH
						* (1.0 / l2Norm(H[curIdx])), H[curIdx]);
				double[] dH = plusVector(dH1, dH2);
				// gradient descent once
				H[curIdx] = plusVector(H[curIdx],
						negVector(dotVector(para.lr, dH)));
			}
			/* update U */
			for (sIdx s : ls) {
				double[] dU1 = dotVector(
						para.lambdaN
								* l2.getPartialDerivation(s.weight,
										predictFeatureF(curIdx, s.sId)),
						H[s.sId]);
				assert (l2Norm(U[curIdx]) != 0);
				double[] dU2 = dotVector(para.lambdaRU
						* (1.0 / l2Norm(U[curIdx])), U[curIdx]);
				double[] dU = plusVector(dU1, dU2);
				// gradient descent once
				U[curIdx] = plusVector(U[curIdx],
						negVector(dotVector(para.lr, dU)));
			}
		}

		@Override
		public void updateH(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update H */
			for (sIdx s : ls) {
				double[] dH1 = dotVector(
						para.lambdaN
								* l2.getPartialDerivation(s.weight,
										predictFeatureF(curIdx, s.sId)),
						U[s.sId]);
				assert (l2Norm(H[curIdx]) != 0);
				double[] dH2 = dotVector(para.lambdaRH
						* (1.0 / l2Norm(H[curIdx])), H[curIdx]);
				double[] dH = plusVector(dH1, dH2);
				// gradient descent once
				H[curIdx] = plusVector(H[curIdx],
						negVector(dotVector(para.lr, dH)));
			}
		}

		@Override
		public void updateU(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub
			/* update U */
			for (sIdx s : ls) {
				double[] dU1 = dotVector(
						para.lambdaN
								* l2.getPartialDerivation(s.weight,
										predictFeatureF(curIdx, s.sId)),
						H[s.sId]);
				assert (l2Norm(U[curIdx]) != 0);
				double[] dU2 = dotVector(para.lambdaRU
						* (1.0 / l2Norm(U[curIdx])), U[curIdx]);
				double[] dU = plusVector(dU1, dU2);
				// gradient descent once
				U[curIdx] = plusVector(U[curIdx],
						negVector(dotVector(para.lr, dU)));
			}
		}

		@Override
		public void updateV(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateZ(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateW(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}

		@Override
		public void updateM(List<sIdx> ls, int curIdx) {
			// TODO Auto-generated method stub

		}
	}

	public void updateAll(List<sIdx>[] container, updateMethod method) {
		for (int i = 0; i < container.length; i++) {
			if (i % 1000 == 0) {
				// System.out.println("update 1000 times.");
			}
			method.update(container[i], i);
		}
	}

	public void updateU2convergence() {
		boolean fail = false;
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		for (int t = 0; t < MRPPara.T; t++) {
			// System.err.println("training U\t"+t);
			for (int i = 0; i < this.featureF.length; i++) {
				uff.updateU(this.featureF[i], i);
				err = this.errFF();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					fail = true;
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
		}
	}

	public void updateV2convergence() {
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		for (int t = 0; t < MRPPara.T; t++) {
			// System.err.println("training U\t"+t);
			for (int i = 0; i < this.GF.length; i++) {
				ugf.updateV(this.GF[i], i);
				err = this.errF();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
			if (t % 10 == 0) {
				// System.out.println("V 10");
			}
		}
	}

	public void updateH2convergence() {
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		for (int t = 0; t < MRPPara.T; t++) {
			for (int i = 0; i < this.GF.length; i++) {
				ugf.updateH(this.GF[i], i);
				err = this.errF();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
			err = 0.0;
			last_err = Double.MAX_VALUE;
			for (int i = 0; i < this.GA.length; i++) {
				uga.updateH(this.GA[i], i);
				err = this.errA();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
			err = 0.0;
			last_err = Double.MAX_VALUE;
			for (int i = 0; i < this.featureF.length; i++) {
				uff.updateH(this.featureF[i], i);
				err = this.errFF();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
		}

	}

	public void updateW2convergence() {
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		for (int t = 0; t < MRPPara.T; t++) {
			for (int i = 0; i < this.GF.length; i++) {
				ugf.updateW(this.GF[i], i);
				err = this.errF();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
			if (t % 10 == 0) {
				// System.out.println("W 10");
			}
		}
	}

	public void updateM2convergence() {
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		for (int t = 0; t < MRPPara.T; t++) {
			for (int i = 0; i < this.GA.length; i++) {
				uga.updateM(this.GA[i], i);
				err = this.errA();
				if ((err <= MRPPara.good) || (err >= last_err)) {
					// System.err.println("line\t" + i);
					break;
				}
				last_err = err;
				// System.out.println(err);
			}
			if (t % 10 == 0) {
				// System.out.println("M 10");
			}
		}
	}

	public void updateZ2convergenceVectorwise() {
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		this.computeFM();
		this.computeHH();
		this.computeYH();
		for (int i = 0; i < dimGA; i++) {
			for (int j = 0; j < dimLatent; j++) {
				this.Z[i][j] = 0;
			}
			uga.updateZVectorwise(this.reverseGA[i], i);
			err = errA();
			if ((err <= MRPPara.good) || (err >= last_err)) {
				// System.err.println("line\t" + t);
				break;
			}
			last_err = err;
			// System.out.println(err);
		}
	}

	public void updateZ2convergence() {
		double err = 0.0;
		double last_err = Double.MAX_VALUE;
		for (int t = 0; t < para.T; t++) {
			for (int i = 0; i < this.GA.length; i++) {
				// for (int i = 0; i < 1; i++) {
				uga.updateZ(this.GA[i], i);
				err = this.errA();
				if ((err <= MRPPara.good) || (err > last_err)) {
					// System.err.println("line\t" + i);
					// break;
				}
				last_err = err;
				System.out.println(err);
			}
		}
	}

	public double errA() {
		double errA = 0.0;
		for (int i = 0; i < GA.length; i++) {
			for (sIdx s : GA[i]) {
				errA += (s.weight - predictAttribute(i, s.sId))
						* (s.weight - predictAttribute(i, s.sId));
			}
		}
		return errA / (dimGA * dimGA);
	}

	public double errF() {
		double errF = 0.0;
		for (int i = 0; i < GF.length; i++) {
			for (sIdx s : GF[i]) {
				double pY = predictFriendship(i, s.sId);
				errF += (s.weight - pY) * (s.weight - pY);
//				if (errF > 1e8) {
//					System.out.println("something wrong.");
//				}
			}
		}
		return errF / (dimGF * dimGF);
	}

	public double errFF() {
		double errFF = 0.0;
		for (int i = 0; i < featureF.length; i++) {
			for (sIdx s : featureF[i]) {
				errFF += (s.weight - predictFeatureF(i, s.sId))
						* (s.weight - predictFeatureF(i, s.sId));
			}
		}
		return errFF / (dimFeatureF * dimFeatureF);
	}

	public double error() {
		double errF = this.errF();
		double errA = this.errA();
		double errFF = this.errFF();
		double err = para.lambdaF * errF + para.lambdaA * errA + para.lambdaN
				* errFF;
		return err;
	}

	public void train(int T) {
		double err = 0.0;
		this.computeReverseGA();
		for (int t = 0; t < T; t++) {
			// fix M, W, H
			System.out.println("U2");
			updateU2convergence();
			System.out.println("V2");
			updateV2convergence();
			// // updateZ2convergence();
			System.out.println("Z2");
			updateZ2convergenceVectorwise();
			// // // fix M, W, U, V, Z
			System.out.println("H2");
			updateH2convergence();
			// // // fix H, U, V, Z
			System.out.println("M2");
			updateM2convergence();
			System.out.println("W2");
			updateW2convergence();
			err = error();
			System.out.println("" + err);
			if (err < 0.01) {
				break;
			}
		}
	}

	public void readFromText(String fGA, String fGF, String fFF, String fFA)
			throws NumberFormatException, IOException {
		readData(fGA, GA, useridx, attridx);
		readData(fGF, GF, useridx, useridx);
		readData(fFF, featureF, useridx, userfidx);
		readData(fFA, featureA, attridx, attrfidx);
		computeReverseGA();
	}

	public void saveParas(String fName, int m, int n, double[][] matrix)
			throws IOException {
		File outf = new File(fName);
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(outf), "utf-8"));
		double tmp = 0.0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n - 1; j++) {
				tmp = matrix[i][j] > 1e-4 ? matrix[i][j] : 0.0;
				writer.write(tmp + ",");
			}
			tmp = matrix[i][n - 1] > 1e-4 ? matrix[i][n - 1] : 0.0;
			writer.write(tmp + "\n");
		}
		writer.close();
	}

	public void saveAll() throws IOException {
		this.saveParas("matrixU", dimGF, dimLatent, this.U);
		this.saveParas("matrixV", dimGF, dimLatent, this.V);
		this.saveParas("matrixH", dimGF, dimLatent, this.H);
		this.saveParas("matrixZ", dimGA, dimLatent, this.Z);
		this.saveParas("matrixW", dimFeatureF, dimFeatureF, this.W);
		this.saveParas("matrixM", dimFeatureF, dimFeatureA, this.M);
	}

	public double MAE(double i, double j) {
		double retval = 0.0;
		retval = Math.abs(i - j);
		return retval;
	}

	public double evaluationFriendship(String fe, int lineCnt)
			throws NumberFormatException, IOException {
		double retval = 0.0;
		List<sIdx>[] le = new ArrayList[lineCnt];
		for (int i = 0; i < lineCnt; i++) {
			le[i] = new ArrayList<sIdx>();
		}
		this.readData(fe, le, this.useridx, this.useridx);
		int cnt = 0;
		for (int i = 0; i < lineCnt; i++) {
			for (sIdx s : le[i]) {
				retval += this.MAE(s.weight, this.predictFriendship(i, s.sId));
				cnt += 1;
			}
		}
		return retval / cnt;
	}

	public double evaluationAttribute(String fe, int lineCnt)
			throws NumberFormatException, IOException {
		double retval = 0.0;
		List<sIdx>[] le = new ArrayList[lineCnt];
		for (int i = 0; i < lineCnt; i++) {
			le[i] = new ArrayList<sIdx>();
		}
		this.readData(fe, le, this.useridx, this.attridx);
		int cnt = 0;
		for (int i = 0; i < lineCnt; i++) {
			for (sIdx s : le[i]) {
				retval += this.MAE(s.weight, this.predictAttribute(i, s.sId));
				cnt += 1;
			}
		}
		return retval;
	}

	/**
	 * @param args
	 * @throws IOException
	 * @throws NumberFormatException
	 */

	public static void main(String[] args) throws NumberFormatException,
			IOException {

		MRP test = new MRP();
		test.init(45683, 1002, 5, 14, 7, 1, 1);
		test.readFromText("final-data/graph-attribute-number.txt",
				"final-data/graph-friendship-number.txt",
				"final-data/feature-user-number.txt",
				"final-data/feature-attribute-number.txt");
		// test.initParameterAndLatentFactor("matrixH", "matrixM", "matrixU",
		// "matrixV", "matrixW", "matrixZ");
		test.initParameterAndLatentFactor();
		test.train(1);
		test.saveAll();
		System.out.println("last error:\t" + test.error());
		// System.out.println("evaluation:\t" +
		// test.evaluationAttribute("C:\\Users\\xusenyin\\Desktop\\weibo_sample\\ga_a0.1",
		// 617));
	}
}
