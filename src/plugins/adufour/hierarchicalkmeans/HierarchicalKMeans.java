package plugins.adufour.hierarchicalkmeans;

import icy.image.IcyBufferedImage;
import icy.image.colormap.FireColorMap;
import icy.main.Icy;
import icy.math.ArrayMath;
import icy.roi.ROI2D;
import icy.roi.ROI2DArea;
import icy.sequence.Sequence;
import icy.swimmingPool.SwimmingObject;
import icy.type.DataType;
import icy.type.collection.array.ArrayUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.vecmath.Point3i;

import plugins.adufour.connectedcomponents.ConnectedComponent;
import plugins.adufour.connectedcomponents.ConnectedComponents;
import plugins.adufour.connectedcomponents.ConnectedComponentsPainter;
import plugins.adufour.ezplug.EzGroup;
import plugins.adufour.ezplug.EzPlug;
import plugins.adufour.ezplug.EzVarBoolean;
import plugins.adufour.ezplug.EzVarDouble;
import plugins.adufour.ezplug.EzVarInteger;
import plugins.adufour.ezplug.EzVarSequence;
import plugins.adufour.filtering.Convolution1D;
import plugins.adufour.filtering.Kernels1D;
import plugins.adufour.thresholder.KMeans;
import plugins.adufour.thresholder.Thresholder;
import plugins.nchenouard.spot.DetectionResult;

public class HierarchicalKMeans extends EzPlug
{
	private EzVarSequence	input;

	private EzVarDouble		preFilterValue;

	private EzVarInteger	minSize;

	private EzVarInteger	maxSize;

	private EzVarInteger	smartLabelClasses;

	private EzVarBoolean	exportSequence, exportSwPool, exportROI;

	@Override
	public void initialize()
	{
		super.addEzComponent(input = new EzVarSequence("Input"));
		super.addEzComponent(preFilterValue = new EzVarDouble("Gaussian pre-filter", 0, 50, 0.1));
		minSize = new EzVarInteger("Min size (px)", 100, 1, 200000000, 1);
		maxSize = new EzVarInteger("Max size (px)", 1600, 1, 200000000, 1);
		addEzComponent(new EzGroup("Object size", minSize, maxSize));

		smartLabelClasses = new EzVarInteger("Number of classes", 10, 2, 255, 1);
		addEzComponent(smartLabelClasses);

		exportSequence = new EzVarBoolean("Labeled sequence", false);
		exportROI = new EzVarBoolean("ROIs", true);
		exportSwPool = new EzVarBoolean("Swimming pool data", false);

		addEzComponent(new EzGroup("Show result as...", exportSequence, exportSwPool, exportROI));
	}

	@Override
	public void execute()
	{
		Sequence labeledSequence = new Sequence();

		Map<Integer, List<ConnectedComponent>> objects = hierarchicalKMeans(input.getValue(), preFilterValue.getValue(),
				smartLabelClasses.getValue(), minSize.getValue(), maxSize.getValue(), labeledSequence);

		// System.out.println("Hierarchical K-Means result:");
		// System.out.println("T\tobjects");
		// for (Integer t : objects.keySet())
		// System.out.println(t + "\t" + objects.get(t).size());
		// System.out.println("---");

		int nbObjects = 0;
		for (List<ConnectedComponent> ccs : objects.values())
			nbObjects += ccs.size();

		if (exportSequence.getValue())
		{
			labeledSequence.addPainter(new ConnectedComponentsPainter(objects));

			labeledSequence.updateComponentsBounds(true, true);
			labeledSequence.getColorModel().setColormap(0, new FireColorMap());

			addSequence(labeledSequence);
		}

		if (exportSwPool.getValue())
		{
			DetectionResult result = ConnectedComponents.convertToDetectionResult(objects, input.getValue());
			SwimmingObject object = new SwimmingObject(result, "Set of " + nbObjects + " connected components");
			Icy.getMainInterface().getSwimmingPool().add(object);
		}

		if (exportROI.getValue() && labeledSequence.getSizeZ() == 1)
		{
			Sequence in = input.getValue();

			in.beginUpdate();

			for (ROI2D roi : input.getValue().getROI2Ds())
				if (roi instanceof ROI2DArea) in.removeROI(roi);

			for (List<ConnectedComponent> ccs : objects.values())
				for (ConnectedComponent cc : ccs)
				{
					ROI2DArea area = new ROI2DArea();
					for (Point3i pt : cc)
						area.addPoint(pt.x, pt.y);
					area.setT(cc.getT());
					in.addROI(area);
				}

			in.endUpdate();
		}

	}

	/**
	 * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
	 * detected objects
	 * 
	 * @param seqIN
	 *            the sequence to segment
	 * @param preFilter
	 *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
	 *            none)
	 * @param nbKMeansClasses
	 *            the number of classes to divide the histogram
	 * @param minSize
	 *            the minimum size in pixels of the objects to segment
	 * @param maxSize
	 *            the maximum size in pixels of the objects to segment
	 * @param seqOUT
	 *            an empty sequence that will receive the labeled output as unsigned short, or null
	 *            if not necessary
	 * @return a map containing the list of connected components found in each time point
	 */
	public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, double preFilter,
			int nbKMeansClasses, int minSize, int maxSize, Sequence seqOUT)
	{
		return hierarchicalKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, null, seqOUT);
	}

	/**
	 * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
	 * detected objects
	 * 
	 * @param seqIN
	 *            the sequence to segment
	 * @param preFilter
	 *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
	 *            none)
	 * @param nbKMeansClasses
	 *            the number of classes to divide the histogram
	 * @param minSize
	 *            the minimum size in pixels of the objects to segment
	 * @param maxSize
	 *            the maximum size in pixels of the objects to segment
	 * @param minValue
	 *            the minimum intensity value each object should have (in any of the input channels)
	 * @param seqOUT
	 *            an empty sequence that will receive the labeled output as unsigned short, or null
	 *            if not necessary
	 * @return a map containing the list of connected components found in each time point
	 */
	public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, double preFilter,
			int nbKMeansClasses, int minSize, int maxSize, Double minValue, Sequence seqOUT)
	{
		if (seqOUT == null) seqOUT = new Sequence();

		seqOUT.setName("Hierarchical KMeans segmentation");

		Sequence seqLABELS = new Sequence();
		Sequence seqC = new Sequence();
		seqC.setName("Current class");

		for (int z = 0; z < seqIN.getSizeZ(); z++)
		{
			seqC.setImage(0, z, new IcyBufferedImage(seqIN.getSizeX(), seqIN.getSizeY(), 1, DataType.UINT));
			seqLABELS.setImage(0, z, new IcyBufferedImage(seqIN.getSizeX(), seqIN.getSizeY(), 1, DataType.USHORT));
		}

		seqOUT.beginUpdate();

		HashMap<Integer, List<ConnectedComponent>> componentsMap = new HashMap<Integer, List<ConnectedComponent>>();

		for (int t = 0; t < seqIN.getSizeT(); t++)
		{
			componentsMap.put(t, new ArrayList<ConnectedComponent>());

			// 1.1) Copy current image in a new sequence

			ArrayUtil.arrayToArray(seqIN.getDataXYZ(t, 0), seqLABELS.getDataXYZ(0, 0), seqIN.getDataType_().isSigned());

			// 1.2) Create the output labeled sequence

			for (int z = 0; z < seqIN.getSizeZ(); z++)
			{
				seqOUT.setImage(t, z, new IcyBufferedImage(seqIN.getSizeX(), seqIN.getSizeY(), 1, DataType.USHORT));
			}

			// 2) Pre-filter the input data

			Kernels1D gaussian = Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter);

			Convolution1D.convolve(seqLABELS, gaussian.getData(), gaussian.getData(), seqIN.getSizeZ() > 1 ? gaussian.getData()
					: null);

			// 3) K-means on the raw data

			Thresholder.threshold(seqLABELS, 0, KMeans.computeKMeansThresholds(seqLABELS, 0, nbKMeansClasses, 255), true);

			// 4) Loop on each class in ascending order

			for (short c = 1; c < nbKMeansClasses; c++)
			{
				// retrieve classes c and above as a binary image
				for (int z = 0; z < seqIN.getSizeZ(); z++)
				{
					short[] _labels = seqLABELS.getDataXYAsShort(0, z, 0);
					int[] _class = seqC.getDataXYAsInt(0, z, 0);
					short[] _out = seqOUT.getDataXYAsShort(t, z, 0);

					for (int i = 0; i < _labels.length; i++)
						if ((_labels[i] & 0xffff) >= c && _out[i] == 0)
						{
							_class[i] = 1;
						}
				}

				// extract connected components on this current class
				{
					Sequence seqLabels = new Sequence();
					List<ConnectedComponent> components = ConnectedComponents.extractConnectedComponents(seqC, minSize,
							maxSize, seqLabels).get(0);
					seqC = seqLabels;

					// assign t value to all components
					for (ConnectedComponent cc : components)
						cc.setT(t);

					if (minValue == null)
					{
						componentsMap.get(t).addAll(components);
					}
					else
					{
						int[][] _class_z_xy = seqC.getDataXYZAsInt(0, 0);

						for (ConnectedComponent cc : components)
						{
							double[] maxIntensities = cc.computeMaxIntensity(seqIN);
							if (ArrayMath.max(maxIntensities) < minValue)
							{
								for (Point3i pt : cc)
								{
									_class_z_xy[pt.z][pt.y * seqC.getSizeX() + pt.x] = 0;
								}
							}
							else
							{
								componentsMap.get(t).add(cc);
							}
						}
					}
				}

				// store the final objects in the output image
				for (int z = 0; z < seqIN.getSizeZ(); z++)
				{
					int[] _class = seqC.getDataXYAsInt(0, z, 0);
					short[] _out = seqOUT.getDataXYAsShort(t, z, 0);

					for (int i = 0; i < _out.length; i++)
					{
						if (_class[i] != 0)
						{
							// store the valid pixel in the output
							_out[i] = 1;
							// erase the pixel from seqC for future classes
							_class[i] = 0;
						}
					}
				}
			}
			System.gc();
		}

		seqOUT.endUpdate();
		seqOUT.dataChanged();
		return componentsMap;
	}

	/**
	 * Performs a hierarchical K-Means segmentation on the input sequence, and returns the result as
	 * a labeled sequence
	 * 
	 * @param seqIN
	 *            the sequence to segment
	 * @param preFilter
	 *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
	 *            none)
	 * @param nbKMeansClasses
	 *            the number of classes to divide the histogram
	 * @param minSize
	 *            the minimum size in pixels of the objects to segment
	 * @param maxSize
	 *            the maximum size in pixels of the objects to segment
	 * @return a labeled sequence with all objects extracted in the different classes
	 */
	public static Sequence hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize)
	{
		return hierarchicalKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, (Double) null);
	}

	/**
	 * Performs a hierarchical K-Means segmentation on the input sequence, and returns the result as
	 * a labeled sequence
	 * 
	 * @param seqIN
	 *            the sequence to segment
	 * @param preFilter
	 *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
	 *            none)
	 * @param nbKMeansClasses
	 *            the number of classes to divide the histogram
	 * @param minSize
	 *            the minimum size in pixels of the objects to segment
	 * @param maxSize
	 *            the maximum size in pixels of the objects to segment
	 * @param minValue
	 *            the minimum intensity value each object should have (in any of the input channels)
	 * @return a labeled sequence with all objects extracted in the different classes
	 */
	public static Sequence hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize,
			Double minValue)
	{
		Sequence result = new Sequence();

		hierarchicalKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, minValue, result);

		return result;
	}

	public void clean()
	{
	}

}
