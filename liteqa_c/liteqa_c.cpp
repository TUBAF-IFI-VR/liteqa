#include <cstddef>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>

////////////////////////////////////////////////////////////////////////
// LITE-QA encoder (LISQ, GLATE, EBATE)

extern "C" {

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// lisq (LOGARITHMIC INCREASING STEPS QUANTIZATION)

#define LISQ_INITIALIZE                                             \
	double log_delta = std::log(delta);                             \
	double log_omega = std::log(1. + omega) - std::log(1. - omega); \
	double div_omega = (1. + omega)/(1. - omega)                    \
//

#define LISQ_INITIALIZE_VEC(N)                                        \
	double log_delta[N];                                              \
	double log_omega[N];                                              \
	double div_omega[N];                                              \
	for (int i = 0; i < (N); i++)                                     \
	{                                                                 \
		double _delta = *(delta + i);                                 \
		double _omega = *(omega + i);                                 \
		log_delta[i] = std::log(_delta);                              \
		log_omega[i] = std::log(1. + _omega) - std::log(1. - _omega); \
		div_omega[i] = (1. + _omega)/(1. - _omega);                   \
	}                                                                 \
//

#define LISQ_STEPNUM(x)                                                                     \
	(x < 0 ? -1:1)*static_cast<int>((std::log(std::fabs((x))) - log_delta)/log_omega + 1.5) \
//

#define LISQ_STEPFUN(n)                                                                               \
	((n) != 0 ? (((n) < 0 ? -1:1)*delta*std::pow(div_omega, std::abs(static_cast<int>(n)) - 1)) : 0.) \
//

#define LISQ_STEPFUN_VEC(i, n)                                                                                            \
	((n) != 0 ? (((n) < 0 ? -1:1)*(*(delta + (i)))*std::pow(*(div_omega + (i)), std::abs(static_cast<int>(n)) - 1)) : 0.) \
//

	float lisq_float_max()
	{
		return std::numeric_limits<float>::max();
	}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// ebate (ERROR-BOUNDED BINNING AND TABULAR ENCODING)

	size_t ebate_encode(
		size_t num,
		double delta,
		double omega,
		const float* data_ptr,
		int* binn_ptr,
		unsigned* bins_ptr,
		unsigned* bini_ptr,
		const unsigned* hmap,
		const unsigned char* mask
	)
	{	// ENCODE INDEX
		LISQ_INITIALIZE;
		std::vector<std::pair<int, int>> step(num);
		int j = 0;
		for (int i = 0; i < num; i++)
		{
			if (*(mask + (*hmap++))) continue;
			++j;
			double x = *(data_ptr + (*(hmap - 1)));
			step[j].second = i;
			if (std::fabs(x) < delta) step[j].first = 0;
			else step[j].first = LISQ_STEPNUM(x);
		}

		//~ printf("%d %d\n", num, j);
		auto step_end = step.begin() + j;
		std::sort(step.begin(), step_end);

		int nbins = 0;
		auto step_ptr = step.begin();
		*binn_ptr++ = step_ptr->first;
		while (true)
		{
			++nbins;
			*bini_ptr++ = step_ptr->second;
			auto step_next = step_ptr + 1;
			//~ printf("%d %d %d\n", step_ptr->first, step_next - step_ptr - 1, step_ptr->second);
			while (step_next->first == step_ptr->first)
			{
				*bini_ptr++ = step_next->second - (step_next - 1)->second - 1;
				//~ printf("%d %d %d\n", step_ptr->first, step_next - step_ptr - 1, step_next->second);
				if (++step_next == step_end) break;
			}
			*bins_ptr++ = step_next - step_ptr;
			if (step_next == step_end) break;
			*binn_ptr++ = step_next->first - step_ptr->first - 1;
			step_ptr = step_next;
		}

		return nbins;
	}

////////////////////////////////////////////////////////////////////////

	void ebate_decode(
		double delta,
		double omega,
		float* data_ptr,
		int nbins,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		const unsigned* bini_ptr,
		const unsigned* hmap
	)
	{	// DECODE INDEX
		LISQ_INITIALIZE;
		int binn = -1;
		for (int i = 0; i < nbins; i++)
		{
			binn += *binn_ptr++ + 1;
			double x = LISQ_STEPFUN(binn);
			unsigned bins = *bins_ptr++;
			unsigned i_last = *bini_ptr++;
			*(data_ptr + *(hmap + i_last)) = x;
			//~ printf("%d %d %d\n", binn, 0, i_last);
			for (int j = 1; j < bins; j++)
			{
				i_last += *bini_ptr++ + 1;
				*(data_ptr + *(hmap + i_last)) = x;
				//~ printf("%d %d %d\n", binn, j, i_last);
			}
		}
	}

////////////////////////////////////////////////////////////////////////

	size_t ebate_count(
		size_t nbin,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		int* min_max,
		int nmin,
		int nmax
	)
	{	// POINT COUNT BY BIN RANGE
		size_t count = 0;
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr++ + 1;
			if (nmin < binn and binn < nmax)
			{
				*(min_max + 0) = std::min(*(min_max + 0), binn);
				*(min_max + 1) = std::max(*(min_max + 1), binn);
				count += *bins_ptr;
			}
			++bins_ptr;
		}
		return count;
	}

	size_t ebate_count_inv(
		size_t nbin,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		int* min_max,
		int nmin,
		int nmax
	)
	{	// POINT COUNT BY BIN RANGE
		size_t count = 0;
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr++ + 1;
			if (binn <= nmin or nmax <= binn)
			{
				*(min_max + 0) = std::min(*(min_max + 0), binn);
				*(min_max + 1) = std::max(*(min_max + 1), binn);
				count += *bins_ptr;
			}
			++bins_ptr;
		}
		return count;
	}

////////////////////////////////////////////////////////////////////////

	size_t ebate_select(
		size_t nbin,
		int* binn_ptr,
		unsigned* bidx_ptr,
		int nmin,
		int nmax
	)
	{	// SELECT INDEX BINS BY RANGE
		size_t bin_sel = 0;
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr + 1;
			*binn_ptr++ = binn;
			if (nmin < binn and binn < nmax)
			{
				*bidx_ptr++ = i;
				++bin_sel;
				//~ printf("%d %d %d\n", nmin, nmax, binn);
			}
		}
		return bin_sel;
	}

	size_t ebate_select_inv(
		size_t nbin,
		int* binn_ptr,
		unsigned* bidx_ptr,
		int nmin,
		int nmax
	)
	{	// SELECT INDEX BINS BY RANGE
		size_t bin_sel = 0;
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr + 1;
			*binn_ptr++ = binn;
			if (binn <= nmin or nmax <= binn)
			{
				*bidx_ptr++ = i;
				++bin_sel;
				//~ printf("%d %d %d\n", nmin, nmax, binn);
			}
		}
		return bin_sel;
	}

////////////////////////////////////////////////////////////////////////

	size_t ebate_merge(
		size_t tab_nbin,
		int* tab_binn_ptr,
		unsigned* tab_bins_ptr,
		size_t nbin,
		const int* binn_ptr,
		const unsigned* bins_ptr
	)
	{	// MERGE INDEX BIN SIZES
		std::vector<int> cpy_binn(tab_binn_ptr, tab_binn_ptr + tab_nbin);
		std::vector<int> cpy_bins(tab_bins_ptr, tab_bins_ptr + tab_nbin);
		tab_nbin = 0;
		auto cpy_binn_ptr = cpy_binn.begin();
		auto cpy_bins_ptr = cpy_bins.begin();
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr++ + 1;
			while (cpy_binn_ptr != cpy_binn.end() and *cpy_binn_ptr < binn)
			{
				*tab_binn_ptr++ = *cpy_binn_ptr++;
				*tab_bins_ptr++ = *cpy_bins_ptr++;
				++tab_nbin;
			}
			if (cpy_binn_ptr != cpy_binn.end() and *cpy_binn_ptr == binn)
			{
				*tab_binn_ptr++ = *cpy_binn_ptr++;
				*tab_bins_ptr = *cpy_bins_ptr++;
				*tab_bins_ptr++ += *bins_ptr++;
				++tab_nbin;
			}
			else
			{
				*tab_binn_ptr++ = binn;
				*tab_bins_ptr++ = *bins_ptr++;
				++tab_nbin;
			}
		}
		while (cpy_binn_ptr != cpy_binn.end())
		{
			*tab_binn_ptr++ = *cpy_binn_ptr++;
			*tab_bins_ptr++ = *cpy_bins_ptr++;
			++tab_nbin;
		}
		return tab_nbin;
	}

////////////////////////////////////////////////////////////////////////

	size_t ebate_filter(
		size_t tab_nbin,
		int* tab_binn_ptr,
		unsigned* tab_bins_ptr,
		size_t nbin,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		int nmin,
		int nmax
	)
	{	// FILTER BIN RANGE AND MERGE INDEX BIN SIZES
		std::vector<int> cpy_binn(tab_binn_ptr, tab_binn_ptr + tab_nbin);
		std::vector<int> cpy_bins(tab_bins_ptr, tab_bins_ptr + tab_nbin);
		tab_nbin = 0;
		auto cpy_binn_ptr = cpy_binn.begin();
		auto cpy_bins_ptr = cpy_bins.begin();
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr++ + 1;
			while (cpy_binn_ptr != cpy_binn.end() and *cpy_binn_ptr < binn)
			{
				if (nmin < *cpy_binn_ptr and *cpy_binn_ptr < nmax)
				{
					*tab_binn_ptr++ = *cpy_binn_ptr++;
					*tab_bins_ptr++ = *cpy_bins_ptr++;
					++tab_nbin;
				}
				else
				{
					++cpy_binn_ptr;
					++cpy_bins_ptr;
				}
			}
			if (cpy_binn_ptr != cpy_binn.end() and *cpy_binn_ptr == binn)
			{
				if (nmin < *cpy_binn_ptr and *cpy_binn_ptr < nmax)
				{
					*tab_binn_ptr++ = *cpy_binn_ptr++;
					*tab_bins_ptr = *cpy_bins_ptr++;
					*tab_bins_ptr++ += *bins_ptr++;
					++tab_nbin;
				}
				else
				{
					++cpy_binn_ptr;
					++cpy_bins_ptr;
					++bins_ptr;
				}
			}
			else
			{
				if (nmin < binn and binn < nmax)
				{
					*tab_binn_ptr++ = binn;
					*tab_bins_ptr++ = *bins_ptr++;
					++tab_nbin;
				}
				else
				{
					++bins_ptr;
				}
			}
		}
		while (cpy_binn_ptr != cpy_binn.end())
		{
			if (nmin < *cpy_binn_ptr and *cpy_binn_ptr < nmax)
			{
				*tab_binn_ptr++ = *cpy_binn_ptr++;
				*tab_bins_ptr++ = *cpy_bins_ptr++;
				++tab_nbin;
			}
			else
			{
				++cpy_binn_ptr;
				++cpy_bins_ptr;
			}
		}
		return tab_nbin;
	}

	size_t ebate_filter_inv(
		size_t tab_nbin,
		int* tab_binn_ptr,
		unsigned* tab_bins_ptr,
		size_t nbin,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		int nmin,
		int nmax
	)
	{	// FILTER BIN RANGE AND MERGE INDEX BIN SIZES
		std::vector<int> cpy_binn(tab_binn_ptr, tab_binn_ptr + tab_nbin);
		std::vector<int> cpy_bins(tab_bins_ptr, tab_bins_ptr + tab_nbin);
		tab_nbin = 0;
		auto cpy_binn_ptr = cpy_binn.begin();
		auto cpy_bins_ptr = cpy_bins.begin();
		int binn = -1;
		for (int i = 0; i < nbin; i++)
		{
			binn += *binn_ptr++ + 1;
			while (cpy_binn_ptr != cpy_binn.end() and *cpy_binn_ptr < binn)
			{
				if (*cpy_binn_ptr <= nmin or nmax <= *cpy_binn_ptr)
				{
					*tab_binn_ptr++ = *cpy_binn_ptr++;
					*tab_bins_ptr++ = *cpy_bins_ptr++;
					++tab_nbin;
				}
				else
				{
					++cpy_binn_ptr;
					++cpy_bins_ptr;
				}
			}
			if (cpy_binn_ptr != cpy_binn.end() and *cpy_binn_ptr == binn)
			{
				if (*cpy_binn_ptr <= nmin or nmax <= *cpy_binn_ptr)
				{
					*tab_binn_ptr++ = *cpy_binn_ptr++;
					*tab_bins_ptr = *cpy_bins_ptr++;
					*tab_bins_ptr++ += *bins_ptr++;
					++tab_nbin;
				}
				else
				{
					++cpy_binn_ptr;
					++cpy_bins_ptr;
					++bins_ptr;
				}
			}
			else
			{
				if (binn <= nmin or nmax <= binn)
				{
					*tab_binn_ptr++ = binn;
					*tab_bins_ptr++ = *bins_ptr++;
					++tab_nbin;
				}
				else
				{
					++bins_ptr;
				}
			}
		}
		while (cpy_binn_ptr != cpy_binn.end())
		{
			if (*cpy_binn_ptr <= nmin or nmax <= *cpy_binn_ptr)
			{
				*tab_binn_ptr++ = *cpy_binn_ptr++;
				*tab_bins_ptr++ = *cpy_bins_ptr++;
				++tab_nbin;
			}
			else
			{
				++cpy_binn_ptr;
				++cpy_bins_ptr;
			}
		}
		return tab_nbin;
	}

////////////////////////////////////////////////////////////////////////

	void ebate_points(
		double delta,
		double omega,
		unsigned cube,
		const unsigned* origin,
		size_t bsel,
		const unsigned* bidx_ptr,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		const unsigned* bini_ptr,
		size_t skip_len,
		unsigned* pts_ptr,
		float* data_ptr,
		const unsigned* hmap
	)
	{	// COMPUTE POINTS FOR INDEX BIN INDICES
		LISQ_INITIALIZE;
		unsigned cube2 = cube*cube;
		pts_ptr += 3*skip_len;
		data_ptr += skip_len;
		int binn = -1;
		for (int j = 0; j < bsel; j++)
		{
			binn += *(binn_ptr + *bidx_ptr) + 1;
			double x = LISQ_STEPFUN(binn);
			int i_last = -1;
			for (int i = 0; i < *(bins_ptr + *bidx_ptr); i++)
			{
				i_last += *bini_ptr++ + 1;
				int pos = *(hmap + i_last);
				unsigned poscube2 = pos % cube2;
				// pos = i + j*n + k*n2
				*pts_ptr++ = *(origin + 0) + (poscube2) % cube;
				*pts_ptr++ = *(origin + 1) + (poscube2) / cube;
				*pts_ptr++ = *(origin + 2) + pos / cube2;
				*data_ptr++ = x;
			}
			++bidx_ptr;
		}
	}

////////////////////////////////////////////////////////////////////////

	size_t ebate_intersect(
		size_t npts,
		unsigned* ptsi_ptr,
		size_t numn,
		size_t maxn,
		int* ptsn_ptr,
		size_t bsel,
		size_t seln,
		const unsigned* bidx_ptr,
		const int* binn_ptr,
		const unsigned* bins_ptr,
		const unsigned* bini_ptr
	)
	{	// INTERSECT POINT INDICES
		std::vector<std::pair<unsigned, int>> data(seln);
		auto data_ptr = data.begin();
		for (int j = 0; j < bsel; j++)
		{
			int binn = *(binn_ptr + *bidx_ptr);
			int i_last = -1;
			for (int i = 0; i < *(bins_ptr + *bidx_ptr); i++)
			{
				i_last += *bini_ptr++ + 1;
				data_ptr->first = i_last;
				data_ptr->second = binn;
				++data_ptr;
			}
			++bidx_ptr;
		}
		std::sort(data.begin(), data_ptr);

		if (numn == 0)
		{
			for (const auto& pair : data)
			{
				*ptsi_ptr++ = pair.first;
				*ptsn_ptr = pair.second;
				ptsn_ptr += maxn;
			}
			return seln;
		}

		unsigned* ptsi_begin = ptsi_ptr;
		unsigned* ptsi_put = ptsi_ptr;
		unsigned* ptsi_end = ptsi_ptr + npts;
		int* ptsn_put = ptsn_ptr;
		unsigned jump = maxn - numn;
		for (const auto& pair : data)
		{
			while (ptsi_ptr != ptsi_end and *ptsi_ptr < pair.first)
			{
				++ptsi_ptr;
				ptsn_ptr += maxn;
			}
			if (ptsi_ptr == ptsi_end) break;
			if (*ptsi_ptr == pair.first)
			{
				*ptsi_put++ = *ptsi_ptr++;
				for (int i = 0; i < numn; i++) *ptsn_put++ = *ptsn_ptr++;
				*ptsn_put = pair.second;
				ptsn_ptr += jump;
				ptsn_put += jump;
			}
		}
		return ptsi_put - ptsi_begin;
	}

////////////////////////////////////////////////////////////////////////

	void ebate_intpoints(
		unsigned cube,
		const unsigned* origin,
		size_t npts,
		const unsigned* ptsi_ptr,
		size_t skip_len,
		unsigned* pts_ptr,
		const unsigned* hmap
	)
	{	// COMPUTE POINTS FOR INDEX BIN INDICES
		unsigned cube2 = cube*cube;
		pts_ptr += 3*skip_len;
		for (int i = 0; i < npts; i++)
		{
			int pos = *(hmap + *ptsi_ptr++);
			unsigned poscube2 = pos % cube2;
			// pos = i + j*n + k*n2
			*pts_ptr++ = *(origin + 0) + (poscube2) % cube;
			*pts_ptr++ = *(origin + 1) + (poscube2) / cube;
			*pts_ptr++ = *(origin + 2) + pos / cube2;
		}
	}

////////////////////////////////////////////////////////////////////////

	void ebate_intdata(
		const double* delta,
		const double* omega,
		size_t npts,
		size_t maxn,
		const int* ptsn_ptr,
		size_t skip_len,
		float* data_ptr
	)
	{	// COMPUTE DATA QUANTIZATION
		LISQ_INITIALIZE_VEC(maxn);
		data_ptr += maxn*skip_len;
		for (int i = 0; i < npts; i++)
		{
			for (unsigned j = 0; j < maxn; j++)
			{
				*data_ptr++ = LISQ_STEPFUN_VEC(j, (*ptsn_ptr++));
			}
		}
	}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// glate (GRID LINEARIZATION AND TABULAR ENCODING)

	void glate_encode(
		size_t num,
		size_t M,
		double delta,
		double omega,
		const float* data_ptr,
		unsigned* expo_ptr,
		unsigned* mant_ptr,
		const unsigned* hmap
	)
	{	// ENCODE STREAM
		LISQ_INITIALIZE;
		unsigned* expo_begin = expo_ptr;
		unsigned* mant_begin = mant_ptr;
		unsigned sign_last = 0;
		unsigned step_last = 0;
		unsigned M_2 = M/2;

		for (int i = 0; i < num; i++)
		{
			double x = *(data_ptr + (*hmap++));
			if (std::fabs(x) < delta)
			{
				*expo_ptr = 0;
				*mant_ptr = 0;
			}
			else
			{

				unsigned n = LISQ_STEPNUM(x);
				int diff = static_cast<int>(n) - static_cast<int>(step_last);
				if (std::abs(diff) < M_2)
				{
					*expo_ptr = 0;
					*mant_ptr = (diff <= 0 ? 1 : 0) + (std::abs(diff) << 1);
					//~ printf("%d %d\n", diff, *mant_ptr);
				}
				else
				{
					*expo_ptr = (n - 1) / M + 1;
					*mant_ptr = (n - 1) % M;
				}

				unsigned sign_this = (x > 0 ? 0 : 1);
				if (sign_this != sign_last) *mant_ptr += M;
				sign_last = sign_this;
				step_last = n;

			}
			//~ printf("%d %d %d %d %d %d %f\n", num, i, *expo_ptr, *mant_ptr, step_last, sign_last, x);
			++expo_ptr;
			++mant_ptr;
		}
	}

////////////////////////////////////////////////////////////////////////

	void glate_decode(
		size_t num,
		size_t M,
		double delta,
		double omega,
		float* data_ptr,
		const unsigned* expo_ptr,
		const unsigned* mant_ptr,
		const unsigned* hmap
	)
	{	// DECODE STREAM
		LISQ_INITIALIZE;
		unsigned sign_last = 0;
		unsigned step_last = 0;
		for (int i = 0; i < num; i++)
		{

			unsigned expo = *expo_ptr++;
			unsigned mant = *mant_ptr++;
			double x = 0;
			if (expo == 0 and mant == 0)
			{
				*(data_ptr + (*hmap++)) = 0;
			}
			else
			{

				if (mant >= M)
				{
					mant -= M;
					sign_last = (sign_last == 1 ? 0 : 1);
				}
				if (expo == 0)
				{
					//~ printf("%d %d\n", mant >> 1, mant);
					if (mant % 2 == 0) step_last += (mant >> 1);
					/***********/ else step_last -= (mant >> 1);
				}
				else
				{
					step_last = (expo - 1)*M + mant + 1;
				}
				*(data_ptr + *(hmap++)) = (sign_last ? -1:1)*LISQ_STEPFUN(step_last);
			}

			//~ printf("%d %d %d %d %d %d %f\n", num, i, expo, mant, step_last, sign_last, x);
		}
	}

////////////////////////////////////////////////////////////////////////

}

////////////////////////////////////////////////////////////////////////

#ifdef LITEQA_C_MAIN

int main()
{
	int E0 = 20;
	int M = 35;

	double omega = std::pow(2., 1./M);
	omega = (omega - 1.)/(omega + 1.);
	double delta = std::pow(2, -E0);
	double log_delta = std::log(delta);
	double log_omega = std::log(1. + omega) - std::log(1. - omega);
	double div_omega = (1. + omega)/(1. - omega);

	float data[] = {0.1, 0.11, 0.111, 0.1111, 0.11111, 0.111111, 0.1111111, 0.11111111, 0.111111111, 0.1111111111};
	unsigned hmap[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	float X[10];
	float Y[10];
	unsigned expo[10];
	unsigned mant[10];

	for (int i = 0; i < 3; i++)
	{
		printf("ITER %d\n", i);
		for (int j = 0; j < 10; j++)
			X[j] = (j % 2 ? -1:1)*data[j]*(i+1);

		glate_encode(10, M, delta, omega, X, expo, mant, hmap);
		glate_decode(10, M, delta, omega, Y, expo, mant, hmap);

		for (int j = 0; j < 10; j++)
		{
			double E = std::fabs(X[j]-Y[j])/std::fabs(X[j])*100;
			printf("%+10.03f %+e %+e %d %d\n", E, X[j], Y[j], expo[j], mant[j]);
		}
	}

}

#endif
