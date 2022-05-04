#include <iostream>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <CImg.h>
#include <chrono>
using namespace cimg_library;


class DisjointSet
{
public:
	DisjointSet(const uint32_t& totalSize) : parents(totalSize)
	{
		for(size_t i = 0; i < totalSize; i++)
			parents[i] = i;
	}

	uint32_t find(uint32_t x)
	{
		uint32_t px;
		while((px = parents[x]) != x)
		{
			uint32_t npx = parents[px];
			parents[x] = npx;
			x = npx;
		}
    	return x;
	}

	void join(uint32_t a, uint32_t b)
	{
		a = find(a);
		b = find(b);
		if(a == b)
			return;

		// prefer linearly increasing labels
		if(b < a)
			std::swap(a, b);

		// update root of b to a
		parents[b] = a;
	}

	friend std::ostream& operator<<(std::ostream& out, DisjointSet& ds)
	{
		for(uint32_t x = 0; x < ds.parents.size(); x++)
		{
			out << ds.find(x);
			if(x < ds.parents.size() - 1)
				out << ", ";
		}
		return out;
	}

private:
	std::vector<uint32_t> parents;
};

// DON'T PROCESS 0s!  Take advantage of vectorized SIMD instructions: requires -march=<your architecture>
// use SZ = 64 (or 32).  If ARM architecture i.e. Jetson, use SZ = 
// NOTE: ptr MUST have SZ elements or more!
static constexpr size_t skipSZ = 32;
static inline bool isAllZeroBlock(const uint8_t* ptr)
{
	using block_t = std::array<uint8_t,skipSZ>;
	block_t data = *(reinterpret_cast<const block_t*>(ptr));

    uint8_t any_true = false;
    for(size_t i = 0; i < skipSZ; i++)
    {
        any_true |= data[i];
    }
    return !any_true;
}

static inline bool isAllOnesBlock(const uint8_t* ptr)
{
	using block_t = std::array<uint8_t,skipSZ>;
	block_t data = *(reinterpret_cast<const block_t*>(ptr));

    uint8_t all_true = true;
    for(size_t i = 0; i < skipSZ; i++)
    {
        all_true &= data[i];
    }
    return all_true;
}

class Segment
{
public:
	uint32_t start;
	uint32_t end;
	uint32_t label;
	Segment() : start(0),end(0),label(0) {}
	Segment(const uint32_t start_, const uint32_t end_, const uint32_t label_) : start(start_),end(end_),label(label_) {}

	bool overlap(const Segment& b) const
	{
		return (start <= b.end && end >= b.start);
	}
};

void getScanlineSegments(const uint8_t* row, const uint32_t width, uint32_t& label, std::vector<Segment>& segments)
{
	Segment segment(0,0,label);
	segments.clear();
	for(uint32_t x = 0; x < width - 1; x++)
	{
		// Actually get the ranges here.
		if(row[x])
		{
			if(segment.end == 0)
				segment.start = x;
			segment.end = x+1;
		}
		else if(segment.end)
		{
			segments.push_back(segment);
			label++;

			segment.start = 0;
			segment.end = 0;
			segment.label = label;
		}

		// 2x speedup for very sparse masks (but not correct):
		const uint8_t* row_x = row + x;
		if(((reinterpret_cast<uintptr_t>(row_x) & skipSZ) == 0) && // If aligned along size block boundary, perform fast check for skipping zeros
			(x < width - skipSZ))
		{
			if(isAllZeroBlock(row_x))
			{
				x += skipSZ;
				continue;
			}
			/*if(isAllOnesBlock(row + x))
			{
				if(segment.end == 0)
					segment.start = x;
				segment.end = x + skipSZ + 1;
				x += skipSZ;
				continue;
			}
			*/
		}

	}	
}

std::vector<std::vector<Segment>> getScanlinesSegments(const uint8_t* ptr, const uint32_t width, const uint32_t height)
{
	std::vector<std::vector<Segment>> scanlineSegments(height);
	std::vector<Segment> segments;
	segments.reserve(8);
	uint32_t label = 0;
	for(uint32_t y = 0; y < height; y++)
	{
		const uint8_t* row = ptr + (y * width);
		getScanlineSegments(row, width, label, segments);
		scanlineSegments[y] = segments;
	}
	return scanlineSegments;
}

class BoundingBox
{
public:
	BoundingBox(const uint32_t top_ = std::numeric_limits<uint32_t>::max(), const uint32_t left_ = std::numeric_limits<uint32_t>::max(), const uint32_t right_ = 0, const uint32_t bottom_ = 0) :
		top(top_), left(left_), right(right_), bottom(bottom_)
	{}
	uint32_t top;		// y coord
	uint32_t left;		// x coord
	uint32_t right;		// x coord
	uint32_t bottom;	// y coord
};

std::unordered_map<uint32_t, BoundingBox> getBoundingBoxes(const std::vector<std::vector<Segment>>& scanlineSegments)
{
	std::unordered_map<uint32_t, BoundingBox> bbs;
	for(size_t y = 0; y < scanlineSegments.size(); y++)
	{
		const std::vector<Segment>& segments = scanlineSegments[y];
		for(const auto& segment : segments)
		{
			uint32_t assignedLabel = segment.label;
			BoundingBox& bb = bbs[assignedLabel];
			if(bb.top > y)
				bb.top = y;
			if(bb.bottom < y)
				bb.bottom = y;
			if(bb.left > segment.start)
				bb.left = segment.start;
			if(bb.right < segment.end)
				bb.right = segment.end;
		}
	}
	return bbs;
}

void draw(CImg<uint32_t>& labels, const std::unordered_map<uint32_t, BoundingBox>& bbs)
{
	const uint32_t bbcolor[] = {static_cast<uint32_t>(bbs.size())};
	for(const std::pair<uint32_t, BoundingBox>& bb : bbs)
	{
		const BoundingBox& labelBB = bb.second;
		const int top = static_cast<int>(labelBB.top);
		const int left = static_cast<int>(labelBB.left);
		const int right = static_cast<int>(labelBB.right);
		const int bottom = static_cast<int>(labelBB.bottom);
		labels.draw_line(left, top, right, top, bbcolor);
		labels.draw_line(right, top, right, bottom, bbcolor);
		labels.draw_line(left, bottom, right, bottom, bbcolor);
		labels.draw_line(left, top, left, bottom, bbcolor);
	}
}

void draw(const std::vector<std::vector<Segment>>& scanlineSegments, const size_t width, const size_t height)
{
	CImg<uint32_t> labels(width, height, 1, 1, 0);
	for(size_t y = 0; y < scanlineSegments.size(); y++)
	{
		const std::vector<Segment>& segments = scanlineSegments[y];
		for(const auto& segment : segments)
		{
			uint32_t assignedLabel = segment.label;
			for(size_t x = segment.start; x < segment.end; x++)
				labels(x,y,0,0) = assignedLabel + 1;
		}
	}
	draw(labels, getBoundingBoxes(scanlineSegments));
	labels.display("labels");
}

void connectedComponents(const CImg<uint8_t>& mask)
{
	std::vector<std::vector<Segment>> scanlineSegments = getScanlinesSegments(mask.data(), mask.width(), mask.height());

	size_t nsegments = 0;
	for(auto& segments : scanlineSegments)
		nsegments += segments.size();
	DisjointSet ds(nsegments);

	for(size_t y = 1; y < scanlineSegments.size(); y++)
	{
		std::vector<Segment>& prevSegments = scanlineSegments[y-1];
		std::vector<Segment>& segments = scanlineSegments[y];
		for(auto& segment : segments)
		{
			for(auto& prevSegment : prevSegments)
			{
				if(segment.overlap(prevSegment))
					ds.join(segment.label, prevSegment.label);
			}
		}
	}

	// Linearize component labels:
	std::unordered_map<uint32_t, uint32_t> remap;
	uint32_t newLabel = 0;
	for(size_t y = 0; y < scanlineSegments.size(); y++)
	{
		std::vector<Segment>& segments = scanlineSegments[y];
		for(auto& segment : segments)
		{
			uint32_t assignedLabel = ds.find(segment.label);
			auto res = remap.emplace(assignedLabel, newLabel);
			if(res.second)
				newLabel++;
			segment.label = res.first->second;
		}
	}
	//draw(scanlineSegments, mask.width(), mask.height());

//	std::unordered_map<uint32_t, BoundingBox> = getBoundingBoxes(scanlineSegments, mask.width(), mask.height());
}

template <typename FUNC>
void benchmark(FUNC f, const size_t niters = 1000)
{
	auto start = std::chrono::high_resolution_clock::now();
	for(size_t iter = 0; iter < niters; iter++)
		f();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Total Duration (s): " << diff.count() / niters << std::endl;
}

void opencvConnectedComponents(const CImg<uint8_t>& mask, CImg<int32_t>& labelImg)
{
	cv::Mat bw(mask.height(), mask.width(), CV_8UC1, const_cast<unsigned char*>(mask.data()));
	cv::Mat labelImage(labelImg.height(), labelImg.width(), CV_32S, labelImg.data());
	int nLabels = cv::connectedComponents(bw, labelImage, 4);
	/*
	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);//background
	for(int label = 1; label < nLabels; ++label)
	{
		colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
	}
	cv::Mat dst(bw.size(), CV_8UC3);
	for(int r = 0; r < dst.rows; ++r)
	{
		for(int c = 0; c < dst.cols; ++c)
		{
			int label = labelImage.at<int>(r, c);
			cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}
	cv::imshow("Connected Components", dst);
	cv::waitKey(0);
	*/
}

CImg<uint8_t> loadMask(const std::string& path = "")
{
	// load trivial mask if no path specified.
	if(path == "")
	{
		size_t w = 1920, h = 1080;
		CImg<uint8_t> mask(w,h,1,1,0);
		for(size_t y = 3; y < 6; y++)
		{
			for(size_t x = 68; x < 72; x++)
				mask(x,y,0,0) = 1;
			for(size_t x = 92; x < 108; x++)
				mask(x,y,0,0) = 1;
		}
		return mask;
	}
	CImg<uint8_t> mask(path.c_str());
	// Ensure binary mask:
	cimg_forXY(mask, x, y)
	{
		if(mask(x,y,0,0) > 1)
			mask(x,y,0,0) = 1;
	}
	mask.resize(mask.width() / 2, mask.height() / 2);
	mask.display("loaded from file");
	return mask;
}

// get scanlines and use disjointset to join labels 

//  g++ -I<path to CImg.h> connected_components.cpp -lpthread -lX11 -march=native -O3 -o connected_components
// OpenCV test:
//  g++ -I/usr/include/opencv4/ -I<path to CImg.h> connected_components.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lpthread -lX11 -O3 -march=native -o connected_components && ./connected_components
int main()
{
	CImg<uint8_t> mask = loadMask("connected_components_test_mask.png");
	bool run_cc_test = true;
	bool run_opencv_test = true;
	bool run_cimg_test = false;
	size_t niters = 2000;

	// 37x faster than CImg .get_label(): Duration (s): 0.000434256
	if(run_cc_test)
	{
		std::cout << "Fast connected components: " << std::endl;
		auto z = [&mask](){ connectedComponents(mask); };
		benchmark(z, niters);
	}

	// OpenCV connected components about 2x as fast: 0.000306708
	if(run_opencv_test)
	{
		std::cout << "Opencv connected components: " << std::endl;
		CImg<int32_t> labelImg(mask.width(),mask.height(),1,1,0);
		auto z3 = [&mask, &labelImg](){ opencvConnectedComponents(mask, labelImg); };
		benchmark(z3,niters);
	}

	// CImg connected components: Duration (s): 0.0211664
	if(run_cimg_test)
	{
		std::cout << "CImg connected components: " << std::endl;
		auto z2 = [&mask](){ mask.get_label(); };
		benchmark(z2, niters);
	}

	//connectedComponents(mask);
	return 0;
}
