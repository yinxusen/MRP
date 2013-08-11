import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class preprocess_weibo {
	DateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	DateFormat sdf2 = new SimpleDateFormat("EEE MMM dd HH:mm:ss 'CST' yyyy",
			Locale.US);
	String formatDate = null;
	Timestamp ts = null;

	public String join(Object[] o, String flag) {
		StringBuffer str_buff = new StringBuffer();
		for (int i = 0, len = o.length; i < len; i++) {
			str_buff.append(String.valueOf(o[i]));
			if (i < len - 1)
				str_buff.append(flag);
		}
		return str_buff.toString();
	}

	public long dateTrans(String str) throws ParseException {
		Date date = sdf2.parse(str);
		return date.getTime();
	}

	public static void main(String[] args) throws IOException, ParseException {
		preprocess_weibo pw = new preprocess_weibo();

		BufferedReader reader = null;
		FileInputStream file = new FileInputStream(new File(
				"/tmp/weibo_yinxusen/weibo.txt"));
		reader = new BufferedReader(new InputStreamReader(file, "UTF-8"));
		String tempString = null;
		File outf = new File("/tmp/weibo_yinxusen/weibo.txt.dateok");
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(outf), "utf-8"));
		int cnt = 0;
		while ((tempString = reader.readLine()) != null) {
			cnt++;
			if (cnt % 1000 == 0) {
				System.out.println("read");
			}
			String[] strArray = tempString.split(",");
			strArray[1] = pw.dateTrans(strArray[1]) + "";
			writer.write(pw.join(strArray, ",")+"\n");
		}
		reader.close();
		writer.close();
	}

}
