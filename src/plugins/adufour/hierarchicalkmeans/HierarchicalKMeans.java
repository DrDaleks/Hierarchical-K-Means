package plugins.adufour.hierarchicalkmeans;

import icy.image.IcyBufferedImage;
import icy.image.colormap.FireColorMap;
import icy.image.colormodel.IcyColorModel;
import icy.main.Icy;
import icy.roi.ROI;
import icy.roi.ROI2D;
import icy.roi.ROI2DArea;
import icy.sequence.DimensionId;
import icy.sequence.Sequence;
import icy.swimmingPool.SwimmingObject;
import icy.system.IcyHandledException;
import icy.type.DataType;
import icy.type.collection.array.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.vecmath.Point3i;

import plugins.adufour.blocks.lang.Block;
import plugins.adufour.blocks.util.VarList;
import plugins.adufour.connectedcomponents.ConnectedComponent;
import plugins.adufour.connectedcomponents.ConnectedComponents;
import plugins.adufour.connectedcomponents.ConnectedComponents.Sorting;
import plugins.adufour.ezplug.EzException;
import plugins.adufour.ezplug.EzGroup;
import plugins.adufour.ezplug.EzLabel;
import plugins.adufour.ezplug.EzPlug;
import plugins.adufour.ezplug.EzVarBoolean;
import plugins.adufour.ezplug.EzVarDimensionPicker;
import plugins.adufour.ezplug.EzVarDouble;
import plugins.adufour.ezplug.EzVarEnum;
import plugins.adufour.ezplug.EzVarInteger;
import plugins.adufour.ezplug.EzVarSequence;
import plugins.adufour.filtering.Convolution1D;
import plugins.adufour.filtering.ConvolutionException;
import plugins.adufour.filtering.Kernels1D;
import plugins.adufour.thresholder.KMeans;
import plugins.adufour.thresholder.Thresholder;
import plugins.adufour.vars.lang.VarGenericArray;
import plugins.adufour.vars.lang.VarROIArray;
import plugins.adufour.vars.lang.VarSequence;
import plugins.adufour.vars.util.VarException;
import plugins.nchenouard.spot.DetectionResult;

public class HierarchicalKMeans extends EzPlug implements Block
{
    protected static int                            resultID          = 1;
    
    protected EzVarSequence                         input             = new EzVarSequence("Input");
    
    protected EzVarDimensionPicker                  channel           = new EzVarDimensionPicker("Channel", DimensionId.C, input.getVariable(), true);
    
    protected EzVarDouble                           preFilterValue    = new EzVarDouble("Gaussian pre-filter", 0, 50, 0.1);
    
    protected EzVarInteger                          minSize           = new EzVarInteger("Min size (px)", 100, 1, 200000000, 1);
    
    protected EzVarInteger                          maxSize           = new EzVarInteger("Max size (px)", 1600, 1, 200000000, 1);
    
    protected EzVarInteger                          smartLabelClasses = new EzVarInteger("Number of classes", 10, 2, 255, 1);
    
    protected EzVarDouble                           finalThreshold    = new EzVarDouble("Final threshold", 0, 0, Short.MAX_VALUE, 1);
    
    protected EzVarBoolean                          exportSequence    = new EzVarBoolean("Labeled sequence", false);
    protected EzVarBoolean                          exportSwPool      = new EzVarBoolean("Swimming pool data", false);
    protected EzVarBoolean                          exportROI         = new EzVarBoolean("ROIs", true);
    
    protected EzVarEnum<Sorting>                    sorting           = new EzVarEnum<ConnectedComponents.Sorting>("Sorting", Sorting.values(), Sorting.DEPTH_ASC);
    
    protected EzLabel                               nbObjects;
    
    protected VarSequence                           outputSequence    = new VarSequence("binary sequence", null);
    
    protected VarGenericArray<ConnectedComponent[]> outputCCs         = new VarGenericArray<ConnectedComponent[]>("objects", ConnectedComponent[].class, null);
    
    protected VarROIArray                           outputROIs        = new VarROIArray("list of ROI");
    
    @Override
    public void initialize()
    {
        super.addEzComponent(input);
        super.addEzComponent(channel);
        channel.setToolTipText("Channel to process (-1 for \"all\")");
        super.addEzComponent(preFilterValue);
        addEzComponent(new EzGroup("Object size", minSize, maxSize));
        addEzComponent(finalThreshold);
        addEzComponent(smartLabelClasses);
        
        addEzComponent(new EzGroup("Show result as...", exportSequence, sorting, exportSwPool, exportROI));
        exportROI.setToolTipText("Create ROIs on the original sequence");
        exportSequence.addVisibilityTriggerTo(sorting, true);
        
        addEzComponent(nbObjects = new EzLabel("< click run to start the detection >"));
    }
    
    @Override
    public void execute()
    {
        Sequence labeledSequence = new Sequence();
        
        Map<Integer, List<ConnectedComponent>> objects = null;
        
        int nbKMeansClasses = smartLabelClasses.getValue();
        if (nbKMeansClasses < 2) throw new VarException("HK-Means requires at least two classes to run");
        
        try
        {
            objects = hierarchicalKMeans(input.getValue(true), channel.getValue(), preFilterValue.getValue(), smartLabelClasses.getValue(), minSize.getValue(), maxSize.getValue(),
                    finalThreshold.getValue(), labeledSequence);
        }
        catch (ConvolutionException e)
        {
            throw new EzException(e.getMessage(), true);
        }
        
        labeledSequence.setName(input.getValue().getName() + "_HK-Means" + (isHeadLess() ? "" : ("#" + resultID++)));
        
        // System.out.println("Hierarchical K-Means result:");
        // System.out.println("T\tobjects");
        int cpt = 0;
        if (objects != null)
        {
            for (Integer t : objects.keySet())
                cpt += objects.get(t).size();
        }
        // System.out.println("---");
        
        if (getUI() != null) nbObjects.setText(cpt + " objects detected");
        
        ArrayList<ConnectedComponent> ccList = new ArrayList<ConnectedComponent>();
        int nbObjects = 0;
        for (List<ConnectedComponent> ccs : objects.values())
        {
            nbObjects += ccs.size();
            ccList.ensureCapacity(nbObjects);
            ccList.addAll(ccs);
        }
        
        outputSequence.setValue(labeledSequence);
        outputCCs.setValue(ccList.toArray(new ConnectedComponent[nbObjects]));
        
        if (exportSequence.getValue())
        {
            ConnectedComponents.createLabeledSequence(labeledSequence, objects, sorting.getValue().comparator);
            labeledSequence.updateChannelsBounds(true);
            
            IcyColorModel cmIN = input.getValue().getColorModel();
            IcyColorModel cmOUT = labeledSequence.getColorModel();
            
            if (channel.getValue() == -1)
            {
                for (int c = 0; c < input.getValue().getSizeC(); c++)
                    cmOUT.setColormap(c, cmIN.getColormap(c));
            }
            else labeledSequence.getColorModel().setColormap(0, new FireColorMap());
            
            addSequence(labeledSequence);
        }
        
        if (exportSwPool.getValue())
        {
            DetectionResult result = ConnectedComponents.convertToDetectionResult(objects, input.getValue());
            SwimmingObject object = new SwimmingObject(result, "Set of " + nbObjects + " connected components");
            Icy.getMainInterface().getSwimmingPool().add(object);
        }
        
        if (exportROI.getValue() || outputROIs.isReferenced())
        {
            if (labeledSequence.getSizeZ() > 1) throw new IcyHandledException("ROI export is not supported in 3D yet.");
            
            ArrayList<ROI2DArea> rois = new ArrayList<ROI2DArea>(objects.size());
            
            cpt = 1;
            for (List<ConnectedComponent> ccs : objects.values())
                for (ConnectedComponent cc : ccs)
                {
                    ROI2DArea area = new ROI2DArea();
                    for (Point3i pt : cc)
                        area.addPoint(pt.x, pt.y);
                    area.setT(cc.getT());
                    area.setName("HK-Means detection #" + cpt++);
                    rois.add(area);
                }
            outputROIs.setValue(rois.toArray(new ROI2DArea[rois.size()]));
            
            if (exportROI.getValue())
            {
                Sequence in = input.getValue();
                
                in.beginUpdate();
                
                for (ROI2D roi : input.getValue().getROI2Ds())
                    if (roi instanceof ROI2DArea) in.removeROI(roi);
                
                for (ROI roi : outputROIs.getValue())
                    in.addROI(roi);
                
                in.endUpdate();
            }
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
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t. the image size
     */
    public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Sequence seqOUT)
            throws ConvolutionException
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
     *            the minimum intensity value each object should have (in its corresponding channel)
     * @param seqOUT
     *            an empty sequence that will receive the labeled output as unsigned short, or null
     *            if not necessary
     * @return a map containing the list of connected components found in each time point
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t the image size
     */
    public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Double minValue, Sequence seqOUT)
            throws ConvolutionException
    {
        return hierarchicalKMeans(seqIN, -1, preFilter, nbKMeansClasses, minSize, maxSize, minValue, seqOUT);
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
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t. the image size
     */
    public static Sequence hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize) throws ConvolutionException
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
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t. the image size
     */
    public static Sequence hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Double minValue) throws ConvolutionException
    {
        Sequence result = new Sequence();
        
        hierarchicalKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, minValue, result);
        
        return result;
    }
    
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
    public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, int channel, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Double minValue,
            Sequence seqOUT) throws ConvolutionException
    {
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
        
        HashMap<Integer, List<ConnectedComponent>> componentsMap = new HashMap<Integer, List<ConnectedComponent>>();
        
        int minC = channel == -1 ? 0 : channel;
        int maxC = channel == -1 ? channels - 1 : channel;
        int sizeC = maxC - minC + 1;
        
        for (int t = 0; t < frames; t++)
        {
            // Create the output labeled sequence
            
            for (int z = 0; z < depth; z++)
                seqOUT.setImage(t, z, new IcyBufferedImage(width, height, sizeC, DataType.UINT));
            
            componentsMap.put(t, new ArrayList<ConnectedComponent>());
            
            for (int c = minC; c <= maxC; c++)
            {
                for (int z = 0; z < depth; z++)
                    Arrays.fill(seqC.getDataXYAsInt(0, z, 0), 0);
                
                // 1) Copy current image in a new sequence
                
                ArrayUtil.arrayToArray(seqIN.getDataXYZ(t, c), seqLABELS.getDataXYZ(0, 0), seqIN.getDataType_().isSigned());
                
                // 2) Pre-filter the input data
                
                double scaleXZ = seqIN.getPixelSizeX() / seqIN.getPixelSizeZ();
                
                Kernels1D gaussianXY = Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter);
                Kernels1D gaussianZ = Kernels1D.CUSTOM_GAUSSIAN.createGaussianKernel1D(preFilter * scaleXZ);
                
                Convolution1D.convolve(seqLABELS, gaussianXY.getData(), gaussianXY.getData(), depth > 1 ? gaussianZ.getData() : null);
                
                // 3) K-means on the raw data
                
                Thresholder.threshold(seqLABELS, 0, KMeans.computeKMeansThresholds(seqLABELS, 0, nbKMeansClasses, 255), true);
                
                // 4) Loop on each class in ascending order
                
                for (short currentClass = 1; currentClass < nbKMeansClasses; currentClass++)
                {
                    // retrieve classes c and above as a binary image
                    for (int z = 0; z < depth; z++)
                    {
                        int[] _labels = seqLABELS.getDataXYAsInt(0, z, 0);
                        int[] _class = seqC.getDataXYAsInt(0, z, 0);
                        int[] _out = seqOUT.getDataXYAsInt(t, z, sizeC == 1 ? 0 : c);
                        
                        for (int i = 0; i < _labels.length; i++)
                            if ((_labels[i] & 0xffff) >= currentClass && _out[i] == 0)
                            {
                                _class[i] = 1;
                            }
                    }
                    
                    // extract connected components on this current class
                    {
                        Sequence seqLabels = new Sequence();
                        List<ConnectedComponent> components = ConnectedComponents.extractConnectedComponents(seqC, minSize, maxSize, seqLabels).get(0);
                        seqC = seqLabels;
                        
                        // assign t/c value to all components
                        for (ConnectedComponent cc : components)
                        {
                            cc.setT(t);
                            cc.setC(c);
                        }
                        
                        if (minValue == null)
                        {
                            componentsMap.get(t).addAll(components);
                        }
                        else
                        {
                            int[][] _class_z_xy = seqC.getDataXYZAsInt(0, 0);
                            
                            for (ConnectedComponent cc : components)
                            {
                                if (cc.computeMaxIntensity(seqIN)[c] < minValue)
                                {
                                    for (Point3i pt : cc)
                                        _class_z_xy[pt.z][pt.y * width + pt.x] = 0;
                                }
                                else
                                {
                                    componentsMap.get(t).add(cc);
                                }
                            }
                        }
                    }
                    
                    // store the final objects in the output image
                    for (int z = 0; z < depth; z++)
                    {
                        int[] _class = seqC.getDataXYAsInt(0, z, 0);
                        int[] _out = seqOUT.getDataXYAsInt(t, z, sizeC == 1 ? 0 : c);
                        
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
            System.gc();
        }
        
        seqOUT.endUpdate();
        seqOUT.dataChanged();
        return componentsMap;
    }
    
    public void clean()
    {
    }
    
    @Override
    public void declareInput(VarList inputMap)
    {
        inputMap.add("Input", input.getVariable());
        inputMap.add("Gaussian pre-filter", preFilterValue.getVariable());
        inputMap.add("Min size (px)", minSize.getVariable());
        inputMap.add("Max size (px)", maxSize.getVariable());
        inputMap.add("Number of classes", smartLabelClasses.getVariable());
        
        // force sequence export in box mode
        exportROI.setValue(false);
        exportSwPool.setValue(false);
        exportSequence.setValue(false);
    }
    
    @Override
    public void declareOutput(VarList outputMap)
    {
        outputMap.add("binary sequence", outputSequence);
        outputMap.add("output objects", outputCCs);
        outputMap.add("output regions", outputROIs);
    }
    
}
