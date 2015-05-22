package plugins.adufour.hierarchicalkmeans;

import icy.image.IcyBufferedImage;
import icy.roi.ROI;
import icy.roi.ROI2D;
import icy.roi.ROI3D;
import icy.roi.ROIUtil;
import icy.sequence.Sequence;
import icy.sequence.SequenceDataIterator;
import icy.type.DataType;
import icy.type.collection.array.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.vecmath.Point3i;

import plugins.adufour.connectedcomponents.ConnectedComponent;
import plugins.adufour.connectedcomponents.ConnectedComponents;
import plugins.adufour.filtering.Convolution1D;
import plugins.adufour.filtering.ConvolutionException;
import plugins.adufour.filtering.Kernels1D;
import plugins.adufour.roi.LabelExtractor;
import plugins.adufour.roi.LabelExtractor.ExtractionType;
import plugins.adufour.thresholder.KMeans;
import plugins.adufour.thresholder.Thresholder;

public class HKMeans
{
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
     * detected objects
     * 
     * @param seqIN
     *            the sequence to segment
     * @param channel
     *            the channel to process (or -1 to process all channels)
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
     * @param channel
     *            the channel to process (or -1 to process all channels)
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
        List<ROI> rois = new ArrayList<ROI>();
        
        Sequence buffer = new Sequence();
        
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
            seqC.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UBYTE));
            seqLABELS.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UBYTE));
        }
        
        for (int z = 0; z < depth; z++)
            buffer.setImage(0, z, new IcyBufferedImage(width, height, 1, DataType.UBYTE));
        
        for (int t = 0; t < frames; t++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int z = 0; z < depth; z++)
                {
                    Arrays.fill(buffer.getDataXYAsByte(0, z, 0), (byte) 0);
                    Arrays.fill(seqC.getDataXYAsByte(0, z, 0), (byte) 0);
                }
                
                // 1) Copy current frame in a new sequence
                
                ArrayUtil.arrayToArray(seqIN.getDataXYZ(t, c), seqLABELS.getDataXYZ(0, 0), seqIN.getDataType_().isSigned());
                
                // 2) apply a multi-class K-means on the raw data
                
                double[] thresholds = KMeans.computeKMeansThresholds(seqLABELS, 0, nbKMeansClasses, 255);
                Thresholder.threshold(seqLABELS, 0, thresholds, true);
                
                // 3) Loop on each class in ascending order
                
                byte currentClass = 1;
                
                while (currentClass < nbKMeansClasses)
                {
                    // retrieve classes c and above as a binary image
                    for (int z = 0; z < depth; z++)
                    {
                        byte[] _labels = seqLABELS.getDataXYAsByte(0, z, 0);
                        byte[] _currentClassMask = seqC.getDataXYAsByte(0, z, 0);
                        byte[] _outputMask = buffer.getDataXYAsByte(0, z, 0);
                        
                        for (int i = 0; i < _labels.length; i++)
                            if (_labels[i] >= currentClass && _outputMask[i] == 0)
                            {
                                _currentClassMask[i] = 1;
                            }
                    }
                    
                    // extract labels on this current class
                    
                    List<ROI> currentROIs = LabelExtractor.extractLabels(seqC, ExtractionType.ANY_LABEL_VS_BACKGROUND, 0);
                    
                    // assign t/c value to all components
                    for (ROI roi : currentROIs)
                    {
                        if (roi instanceof ROI2D)
                        {
                            ((ROI2D) roi).setT(t);
                            ((ROI2D) roi).setC(c);
                        }
                        else if (roi instanceof ROI3D)
                        {
                            ((ROI3D) roi).setT(t);
                            ((ROI3D) roi).setC(c);
                        }
                    }
                    
                    if (minIntensity == null)
                    {
                        rois.addAll(currentROIs);
                    }
                    else
                    {
                        byte[][] _class_z_xy = seqC.getDataXYZAsByte(0, 0);
                        
                        for (ROI roi : currentROIs)
                        {
                            if (ROIUtil.getMaxIntensity(seqIN, roi) < minIntensity)
                            {
                                SequenceDataIterator it = new SequenceDataIterator(seqC, roi, true);
                                
                                while (!it.done())
                                {
                                    _class_z_xy[it.getPositionZ()][it.getPositionY() * width + it.getPositionX()] = 0;
                                }
                            }
                            else
                            {
                                rois.add(roi);
                            }
                        }
                    }
                    
                    // store the final objects in the output image
                    for (int z = 0; z < depth; z++)
                    {
                        byte[] _class = seqC.getDataXYAsByte(0, z, 0);
                        byte[] _out = buffer.getDataXYAsByte(0, z, c);
                        
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
        
        return rois;
    }
    
}
