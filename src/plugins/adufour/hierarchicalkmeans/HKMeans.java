package plugins.adufour.hierarchicalkmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.vecmath.Point3i;

import icy.image.IcyBufferedImage;
import icy.roi.ROI;
import icy.roi.ROI2D;
import icy.roi.ROI3D;
import icy.sequence.Sequence;
import icy.sequence.SequenceDataIterator;
import icy.type.DataIteratorUtil;
import icy.type.DataType;
import icy.type.collection.array.Array1DUtil;
import icy.type.collection.array.ArrayUtil;
import plugins.adufour.connectedcomponents.ConnectedComponent;
import plugins.adufour.connectedcomponents.ConnectedComponents;
import plugins.adufour.ezplug.EzStatus;
import plugins.adufour.filtering.Convolution1D;
import plugins.adufour.filtering.ConvolutionException;
import plugins.adufour.filtering.Kernels1D;
import plugins.adufour.roi.LabelExtractor;
import plugins.adufour.roi.LabelExtractor.ExtractionType;
import plugins.adufour.thresholder.KMeans;
import plugins.adufour.thresholder.Thresholder;
import plugins.kernel.roi.descriptor.intensity.ROIMaxIntensityDescriptor;

/**
 * Extracts objects based on multiple thresholds and size constraints
 * 
 * @author Alexandre Dufour
 */
public class HKMeans
{
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
     *            the minimum intensity value each object should have (in its corresponding channel)
     * @param seqOUT
     *            an empty sequence that will receive the labeled output as unsigned short, or null
     *            if not necessary
     * @return a map containing the list of connected components found in each time point
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t the image size
     * @deprecated ConnectedComponent objects are deprecated. Use
     *             {@link #hKMeans(Sequence, byte, int, int, Double)} instead
     */
    public static List<ConnectedComponent> hKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Double minValue, Sequence seqOUT)
            throws ConvolutionException
    {
        boolean outputLabeledSequence = (seqOUT != null);
        if (seqOUT == null) seqOUT = new Sequence();
        
        final int width = seqIN.getSizeX();
        final int height = seqIN.getSizeY();
        final int depth = seqIN.getSizeZ();
        final int frames = seqIN.getSizeT();
        final int channels = seqIN.getSizeC();
        
        Sequence seqLABELS = new Sequence();
        Sequence seqC = new Sequence();
        seqC.setName("Current class");
        
        for (int z = 0; z < depth; z++)
        {
            seqC.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UINT));
            seqLABELS.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UINT));
        }
        
        seqOUT.beginUpdate();
        
        List<ConnectedComponent> components = new ArrayList<ConnectedComponent>();
        
        if (outputLabeledSequence)
        {
            for (int t = 0; t < frames; t++)
                for (int z = 0; z < depth; z++)
                    seqOUT.setImage(t, z, new IcyBufferedImage(width, height, channels, DataType.UINT));
        }
        else
        {
            for (int z = 0; z < depth; z++)
                seqOUT.setImage(0, z, new IcyBufferedImage(width, height, channels, DataType.UINT));
        }
        
        for (int t = 0; t < frames; t++)
        {
            int outT = (outputLabeledSequence ? t : 0);
            
            for (int c = 0; c < channels; c++)
            {
                for (int z = 0; z < depth; z++)
                {
                    Arrays.fill(seqC.getDataXYAsInt(0, z, 0), 0);
                    if (outT < t) Arrays.fill(seqOUT.getDataXYAsInt(0, z, 0), 0);
                }
                
                // 1) Copy current image in a new sequence
                
                ArrayUtil.arrayToArray(seqIN.getDataXYZ(t, c), seqLABELS.getDataXYZ(0, 0), seqIN.getDataType_().isSigned());
                
                // 2) Pre-filter the input data
                
                double scaleXZ = seqIN.getPixelSizeX() / seqIN.getPixelSizeZ();
                
                if (preFilter > 0)
                {
                    Kernels1D gaussianXY = Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter);
                    Kernels1D gaussianZ = Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter * scaleXZ);
                    Convolution1D.convolve(seqLABELS, gaussianXY.getData(), gaussianXY.getData(), depth > 1 ? gaussianZ.getData() : null);
                }
                if (Thread.currentThread().isInterrupted())
                {
                    System.out.println("[HK-Means] Process interrupted");
                    return components;
                }
                
                // 3) K-means on the raw data
                
                Thresholder.threshold(seqLABELS, 0, KMeans.computeKMeansThresholds(seqLABELS, 0, nbKMeansClasses, 255), true);
                
                // 4) Loop on each class in ascending order
                
                for (short currentClass = 1; currentClass < nbKMeansClasses; currentClass++)
                {
                    if (Thread.currentThread().isInterrupted())
                    {
                        System.out.println("[HK-Means] Process interrupted");
                        return components;
                    }
                    
                    // retrieve classes c and above as a binary image
                    for (int z = 0; z < depth; z++)
                    {
                        int[] _labels = seqLABELS.getDataXYAsInt(0, z, 0);
                        int[] _class = seqC.getDataXYAsInt(0, z, 0);
                        int[] _out = seqOUT.getDataXYAsInt(outT, z, c);
                        
                        for (int i = 0; i < _labels.length; i++)
                            if ((_labels[i] & 0xffff) >= currentClass && _out[i] == 0)
                            {
                                _class[i] = 1;
                            }
                    }
                    
                    // extract connected components on this current class
                    {
                        Sequence seqLabels = new Sequence();
                        List<ConnectedComponent> currentCC = ConnectedComponents.extractConnectedComponents(seqC, minSize, maxSize, seqLabels).get(0);
                        seqC = seqLabels;
                        
                        // assign t/c value to all components
                        for (ConnectedComponent cc : currentCC)
                        {
                            cc.setT(t);
                            cc.setC(c);
                        }
                        
                        if (minValue == null)
                        {
                            components.addAll(currentCC);
                        }
                        else
                        {
                            int[][] _class_z_xy = seqC.getDataXYZAsInt(0, 0);
                            
                            for (ConnectedComponent cc : currentCC)
                            {
                                if (cc.computeMaxIntensity(seqIN)[c] < minValue)
                                {
                                    for (Point3i pt : cc)
                                        _class_z_xy[pt.z][pt.y * width + pt.x] = 0;
                                }
                                else
                                {
                                    components.add(cc);
                                }
                            }
                        }
                    }
                    
                    // store the final objects in the output image
                    for (int z = 0; z < depth; z++)
                    {
                        int[] _class = seqC.getDataXYAsInt(0, z, 0);
                        int[] _out = seqOUT.getDataXYAsInt(outT, z, c);
                        
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
                } // currentClass
            }
        }
        
        seqOUT.endUpdate();
        seqOUT.dataChanged();
        return components;
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
     * detected objects
     * 
     * @param seqIN
     *            the sequence to segment
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram (up to 255)
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @param minIntensity
     *            the minimum intensity value each object should have (in its corresponding channel)
     * @return a list of ROI extracted from the input sequence
     */
    public static List<ROI> hKMeans(Sequence seqIN, byte nbKMeansClasses, int minSize, int maxSize, Double minIntensity)
    {
        return hKMeans(seqIN, -1, -1, 0.0, nbKMeansClasses, minSize, maxSize, minIntensity, (EzStatus) null);
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
     * detected objects
     * 
     * @param seqIN
     *            the sequence to segment
     * @param t
     *            the time point to process (or -1 to process all time points)
     * @param c
     *            the channel to process (or -1 to process all channels)
     * @param preFilter
     *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
     *            none)
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram (up to 255)
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @param minIntensity
     *            the minimum intensity value each object should have (in its corresponding channel)
     * @param status
     *            an {@link EzStatus} object to monitor the task progression (or <code>null</code>
     *            if not available or not needed)
     * @return a list of ROI extracted from the input sequence
     */
    public static List<ROI> hKMeans(Sequence seqIN, int t, int c, double preFilter, byte nbKMeansClasses, int minSize, int maxSize, Double minIntensity, EzStatus status)
    {
        List<ROI> rois = new ArrayList<ROI>();
        
        int minT = t >= 0 ? t : 0, maxT = t >= 0 ? t : seqIN.getSizeT() - 1;
        int minC = c >= 0 ? c : 0, maxC = c >= 0 ? c : seqIN.getSizeC() - 1;
        
        final int width = seqIN.getSizeX();
        final int height = seqIN.getSizeY();
        final int depth = seqIN.getSizeZ();
        final DataType dataType = seqIN.getDataType_();
        
        // Expected memory overhead (in bytes): 3 x width x height x depth
        Sequence allClasses = new Sequence("Labels in " + seqIN.getName());
        Sequence currentClass = new Sequence("Current class");
        Sequence finalBinaryOutput = new Sequence("Objects found in " + seqIN.getName());
        
        for (int z = 0; z < depth; z++)
        {
            currentClass.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UBYTE));
            allClasses.setImage(0, z, new IcyBufferedImage(width, height, 1, dataType));
            finalBinaryOutput.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UBYTE));
        }
        // NB: The final (binary) output could be given to the user in addition to the extracted ROI
        
        for (t = minT; t <= maxT; t++)
        {
            if (status != null && maxT - minT > 0)
            {
                status.setMessage("Processing T=" + t);
                status.setCompletion((t + 1) / (double) (maxT - minT + 1));
            }
            
            for (c = minC; c <= maxC; c++)
            {
                if (status != null && maxC - minC > 0)
                {
                    status.setMessage("Processing T=" + t + ", C=" + c);
                }
                
                // 0) reset temporary buffers if necessary
                if (t > minT || c > minC) for (int z = 0; z < depth; z++)
                {
                    Arrays.fill(finalBinaryOutput.getDataXYAsByte(0, z, 0), (byte) 0);
                    Arrays.fill(currentClass.getDataXYAsByte(0, z, 0), (byte) 0);
                }
                
                // 1) Copy current frame in a new sequence
                
                ArrayUtil.arrayToArray(seqIN.getDataXYZ(t, c), allClasses.getDataXYZ(0, 0), dataType.isSigned());
                
                // 2) Gaussian filtering
                
                if (preFilter > 0) try
                {
                    double scaleXZ = seqIN.getPixelSizeX() / seqIN.getPixelSizeZ();
                    Sequence gaussianXY = Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter).toSequence();
                    Sequence gaussianZ = depth == 1 ? null : Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter * scaleXZ).toSequence();
                    Convolution1D.convolve(allClasses, gaussianXY, gaussianXY, depth > 1 ? gaussianZ : null);
                }
                catch (ConvolutionException e)
                {
                    System.err.println("[HK-Means] Warning: couldn't pre-filter. Skipping...");
                }
                
                // 2) apply a multi-class K-means on the raw data
                
                double[] thresholds = KMeans.computeKMeansThresholds(allClasses, 0, nbKMeansClasses, 255);
                Thresholder.threshold(allClasses, 0, thresholds, true);
                
                // 3) Loop on each class in ascending order
                
                for (short currentClassID = 1; currentClassID < nbKMeansClasses; currentClassID++)
                {
                    if (Thread.currentThread().isInterrupted()) return rois;
                    
                    // 3.a) retrieve classes c and above as a binary image
                    // (except where objects have already been found)
                    
                    for (int z = 0; z < depth; z++)
                    {
                        Object _allClasses = allClasses.getDataXY(0, z, 0);
                        byte[] _currentClass = currentClass.getDataXYAsByte(0, z, 0);
                        byte[] _outputMask = finalBinaryOutput.getDataXYAsByte(0, z, 0);
                        
                        int offset = 0;
                        for (int j = 0; j < height; j++)
                            for (int i = 0; i < width; i++, offset++)
                                if (_outputMask[offset] == 0 && Array1DUtil.getValue(_allClasses, offset, dataType) >= currentClassID)
                                {
                                    _currentClass[offset] = 1;
                                }
                                else
                                {
                                    _currentClass[offset] = 0;
                                }
                    }
                    
                    // 3.b) extract labels on this current class
                    
                    List<ROI> currentROIs = LabelExtractor.extractLabelsSlower(currentClass, 0, 0, ExtractionType.ANY_LABEL_VS_BACKGROUND, 0);
                    
                    // Discard ROIs violating the size or intensity constraints
                    for (int i = 0; i < currentROIs.size(); i++)
                    {
                        ROI currentROI = currentROIs.get(i);
                        
                        double size = currentROI.getNumberOfPoints();
                        if (size < minSize || size > maxSize)
                        {
                            currentROIs.remove(i--);
                            continue;
                        }
                        
                        double maxIntensity = ROIMaxIntensityDescriptor.computeMaxIntensity(currentROI, seqIN);
                        
                        if (minIntensity != null && maxIntensity < minIntensity)
                        {
                            currentROIs.remove(i--);
                            continue;
                        }
                    }
                    
                    if (currentROIs.isEmpty()) continue;
                    
                    // All remaining ROIs are now valid
                    rois.addAll(currentROIs);
                    
                    // store the final objects in the output image
                    for (ROI currentROI : currentROIs)
                        DataIteratorUtil.set(new SequenceDataIterator(finalBinaryOutput, currentROI), 1);
                        
                    // Finally, set the proper T / C and color
                    for (ROI currentROI : currentROIs)
                    {
                        if (currentROI instanceof ROI2D)
                        {
                            ((ROI2D) currentROI).setC(c);
                            ((ROI2D) currentROI).setT(t);
                        }
                        else if (currentROI instanceof ROI3D)
                        {
                            ((ROI2D) currentROI).setC(c);
                            ((ROI2D) currentROI).setT(t);
                        }
                        currentROI.setColor(seqIN.getColorMap(c).getDominantColor().brighter());
                    }
                    
                } // currentClass
            }
        }
        
        return rois;
    }
    
}
