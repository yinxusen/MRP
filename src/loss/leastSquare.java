package loss;

public class leastSquare implements loss {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	leastSquare() {

	}

	private static final leastSquare m_instance = new leastSquare();

	public static leastSquare getInstance() {
		return m_instance;
	}

	public double getLoss(double a, double b) {
		return (a - b) * (a - b);
	}

	public double getPartialDerivation(double a, double b) {
		return -2 * (a - b);
	}

}
